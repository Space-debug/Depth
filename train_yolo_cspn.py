#!/usr/bin/env python3
"""CSPN-guided refinement for YOLO26-Depth (Ultralytics 8.4.x), third-party repos untouched.

Architecture (integrates INTO the DPT-style depth head, CSPN attached after it):

    RGB ─► YOLO26 backbone/FPN ─► Depth head (pretrained, reused layer-by-layer)
                                    ├─ coarse depth  (B,1,H/4,W/4), exp() metric head
                                    └─ guidance 8ch  (NEW branch on the fused features)
                                              └──► CSPN (frozen, 24 steps) ──► refined depth

The wrapped head returns {"depth": refined} in training mode, so the stock Ultralytics
DepthLoss26 / DepthValidator / checkpointing supervise and evaluate the REFINED output
unchanged. Sparse-depth pinning (depth-completion mode) samples --n-sample points from
GT during training; with --n-sample 0 it is pure monocular refinement.

Two-stage recipe (recommended):
  Stage 1 (train guidance branch only):
    python train_yolo_cspn.py --model yolo26n-depth.pt --data nyu-depth.yaml \
        --epochs 5 --train-scope guide --lr0 1e-4 --name stage1
  Stage 2 (joint fine-tune, small lr to avoid forgetting):
    python train_yolo_cspn.py --model runs/depth/stage1/weights/best.pt --data nyu-depth.yaml \
        --epochs 20 --train-scope all --lr0 1e-5 --name stage2

Inference (same script so pickled classes resolve):
    python train_yolo_cspn.py --predict --weights runs/depth/stage2/weights/best.pt --source img.jpg

Notes
-----
- Requires the folder layout: ./ultralytics (source repo) and ./CSPN/cspn_pytorch (CSPN repo).
- Checkpoints pickle the wrapper classes defined HERE; to load them elsewhere do
  `import train_yolo_cspn` first (or run inference through this script).
- Single GPU assumed; CSPN refinement runs at head resolution (input/4), where 24 steps
  give an effective propagation radius of ~96 px at full resolution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in (ROOT / "ultralytics", ROOT / "CSPN" / "cspn_pytorch" / "models"):
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.depth.train import DepthTrainer
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DepthModel, load_checkpoint
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import intersect_dicts

from cspn import Affinity_Propagate  # CSPN repo, imported as a top-level module


# --------------------------------------------------------------------------------------
# CSPN: subclass reusing the repo's affinity_normalization / pad_blur_depth as-is; only the
# driver loop is rewritten so the fixed all-ones sum conv is registered once in __init__
# (the original recreates it and calls .cuda() inside forward on every call).
# --------------------------------------------------------------------------------------
class CSPN(Affinity_Propagate):
    """Frozen CSPN propagation (zero learnable parameters)."""

    def __init__(self, prop_time=24, prop_kernel=3, norm_type="8sum", pin_to="sparse"):
        super().__init__(prop_time, prop_kernel, norm_type)
        self.pin_to = pin_to
        self.sum_conv = nn.Conv3d(8, 1, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
        self.sum_conv.weight.data.fill_(1.0)
        self.sum_conv.weight.requires_grad = False

    def _apply(self, fn):
        # EMA/validation models get .half()'d (trainer.py ModelEMA); forward() runs the
        # propagation in fp32, so keep the fixed all-ones sum conv aligned with that.
        ret = super()._apply(fn)
        self.sum_conv.weight.data = self.sum_conv.weight.data.float()
        return ret

    def forward(self, guidance, blur_depth, sparse_depth=None):
        # fp16 (AMP) underflows the affinity normalization division; run propagation in fp32.
        # The refinement is cheap (H/4 resolution), so the precision restore costs nothing.
        with torch.autocast(device_type=guidance.device.type, enabled=False):
            guidance = guidance.float()
            blur_depth = blur_depth.float()
            sparse_depth = sparse_depth.float() if sparse_depth is not None else None
            gate_wb, gate_sum = self.affinity_normalization(guidance)  # inherited
            result = blur_depth
            sparse_mask = sparse_depth.sign() if sparse_depth is not None else None
            for _ in range(self.prop_time):
                stacked = self.pad_blur_depth(result)  # neighbor-aligned copies, (B,1,8,H+2,W+2)
                neighbor_sum = self.sum_conv(gate_wb * stacked).squeeze(1)[:, :, 1:-1, 1:-1]
                result = (1.0 - gate_sum) * blur_depth + neighbor_sum
                if sparse_mask is not None:
                    anchor = sparse_depth if self.pin_to == "sparse" else blur_depth
                    result = (1 - sparse_mask) * result + sparse_mask * anchor
        return result


class _SparseState:
    """Thread-simple holder letting the model smuggle sampled sparse points to the head."""

    def __init__(self):
        self.tensor = None


# --------------------------------------------------------------------------------------
# Wrapped depth head: runs the ORIGINAL Depth head layers (pretrained weights reused),
# taps the fused features for a guidance branch, and refines with CSPN.
# --------------------------------------------------------------------------------------
class CSPNDepthHead(nn.Module):
    """Drop-in replacement for ultralytics Depth head; keeps its train/eval/export contract."""

    def __init__(self, inner: nn.Module, cspn_steps=24, norm_type="8sum", sparse_state=None):
        super().__init__()
        self.inner = inner  # the original Depth head instance (weights preserved)
        self.i = getattr(inner, "i", -1)  # graph attrs required by _predict_once
        self.f = getattr(inner, "f", -1)
        self.sparse_state = sparse_state

        c_mid = inner.proj[0].conv.out_channels  # fusion channel dim (256 by default)
        self.guide = nn.Sequential(  # fused feats (H/8) -> 8ch affinity (H/4)
            Conv(c_mid, c_mid // 4, k=3),
            nn.ConvTranspose2d(c_mid // 4, c_mid // 4, kernel_size=2, stride=2),
            nn.Conv2d(c_mid // 4, 8, kernel_size=3, padding=1),  # no act: signed affinity
        )
        self.cspn = CSPN(cspn_steps, 3, norm_type=norm_type)

    def _current_sparse(self, guidance):
        s = self.sparse_state.tensor if self.sparse_state is not None else None
        if s is None:
            return None
        s = s.to(guidance.device, guidance.dtype).unsqueeze(1)  # (B,1,H,W) meters, 0 elsewhere
        return F.interpolate(s, size=guidance.shape[-2:], mode="nearest")

    def forward(self, x):
        inner = self.inner
        feats = [inner.proj[i](x[i]) for i in range(inner.nl)]  # same fusion as Depth.forward
        out = feats[-1]
        for i in range(inner.nl - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
            out = out + feats[i]
            out = inner.refine[i](out)
        coarse = torch.exp(inner.head(out).clamp(-4.0, 5.0))  # (B,1,H/4,W/4) metric depth
        guidance = self.guide(out)  # (B,8,H/4,W/4)
        refined = self.cspn(guidance, coarse, self._current_sparse(guidance))

        if self.training:
            return {"depth": refined}  # DepthLoss26 supervises the refined output
        depth = refined.pow(inner.cal_a) * inner.cal_b.exp()
        if inner.export:
            depth = F.interpolate(depth, scale_factor=4.0, mode="bilinear", align_corners=False)
        return depth


# --------------------------------------------------------------------------------------
# Model: swap the head at construction time; remap checkpoint keys; sample sparse in loss().
# --------------------------------------------------------------------------------------
class CSPNDepthModel(DepthModel):
    def __init__(self, cfg="yolo26n-depth.yaml", ch=3, nc=None, verbose=True,
                 cspn_steps=24, norm_type="8sum", n_sample=0):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.n_sample = n_sample
        self._sparse_state = _SparseState()
        self.model[-1] = CSPNDepthHead(self.model[-1], cspn_steps, norm_type, self._sparse_state)

    def load(self, weights, verbose=True):
        """Load official yolo26*-depth.pt (keys model.N.*) or our own checkpoints (model.N.inner.*).

        `weights` may be an nn.Module (what the trainer passes), a checkpoint dict, or a path.
        """
        if isinstance(weights, (str, Path)):
            weights, _ = load_checkpoint(str(weights))
        model = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        head_idx = len(self.model) - 1
        prefix, inner_prefix = f"model.{head_idx}.", f"model.{head_idx}.inner."
        csd = {inner_prefix + k[len(prefix):] if k.startswith(prefix) and ".inner." not in k else k: v
               for k, v in csd.items()}
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)
        if verbose:
            print(f"CSPNDepthModel.load: transferred {len(csd)}/{len(self.state_dict())} tensors")

    @staticmethod
    def _sample_sparse(gt: torch.Tensor, n_sample: int) -> torch.Tensor:
        """Randomly keep n_sample valid GT pixels per image (depth-completion simulation)."""
        gt = gt.squeeze(1) if gt.ndim == 4 else gt  # trainer batches carry (B,1,H,W)
        sparse = torch.zeros_like(gt)
        for b in range(gt.shape[0]):
            ys, xs = torch.nonzero(gt[b] > 0.001, as_tuple=True)
            if ys.numel() == 0:
                continue
            sel = torch.randperm(ys.numel(), device=gt.device)[: min(n_sample, ys.numel())]
            sparse[b, ys[sel], xs[sel]] = gt[b, ys[sel], xs[sel]]
        return sparse

    def loss(self, batch, preds=None):
        gt = batch.get("depth")
        if self.training and self.n_sample > 0 and gt is not None:
            self._sparse_state.tensor = self._sample_sparse(gt.float(), self.n_sample)
        try:
            return super().loss(batch, preds)
        finally:
            self._sparse_state.tensor = None


# --------------------------------------------------------------------------------------
# Trainer: stock DepthTrainer, only model construction is customized.
# --------------------------------------------------------------------------------------
class CSPNDepthTrainer(DepthTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None,
                 cspn_steps=24, norm_type="8sum", n_sample=0, train_scope="guide"):
        self.cspn_steps, self.norm_type = cspn_steps, norm_type
        self.n_sample, self.train_scope = n_sample, train_scope
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CSPNDepthModel(
            cfg or "yolo26n-depth.yaml",
            ch=self.data.get("channels", 3),
            nc=self.data["nc"],
            verbose=verbose,
            cspn_steps=self.cspn_steps,
            norm_type=self.norm_type,
            n_sample=self.n_sample,
        )
        if weights:
            model.load(weights)
        return model

    def setup_model(self):
        """Configure freezing through the stock `args.freeze` mechanism (name-substring matching).

        The stock trainer force-resets any requires_grad=False param that is NOT covered by its
        freeze list (see BaseTrainer._setup_train), so manual freezing in get_model() is undone.
        Expressing scopes as freeze names survives that loop and additionally freezes BatchNorm
        running stats of frozen layers (trainer._model_train sets their BN back to eval each epoch).

        freeze list entries are rendered as f"model.{x}." and matched by substring:
          guide: ["0".."N-1", "N.inner", "N.cspn"] -> only model.N.guide.* stays trainable
          head:  ["0".."N-1", "N.cspn"]            -> backbone/neck frozen, whole head trains
          all:   ["N.cspn"]                        -> joint fine-tune (CSPN sum conv stays frozen)
        """
        ckpt = super().setup_model()
        # channels_last is incompatible with the CSPN 5-D sum conv; disable it (negligible for a 5M model)
        self.args.channels_last = False
        last = len(self.model.model) - 1  # head layer index, e.g. 23
        prefix = [str(i) for i in range(last)]
        if self.train_scope == "guide":
            self.args.freeze = prefix + [f"{last}.inner", f"{last}.cspn"]
        elif self.train_scope == "head":
            self.args.freeze = prefix + [f"{last}.cspn"]
        else:  # all
            self.args.freeze = [f"{last}.cspn"]  # keep the fixed all-ones sum conv non-trainable
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"train_scope={self.train_scope!r}: {n_train / 1e6:.2f}M params initially trainable")
        return ckpt


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def train(args):
    overrides = dict(model=args.model, data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                     batch=args.batch, device=args.device, workers=args.workers, lr0=args.lr0,
                     name=args.name, exist_ok=args.exist_ok)
    if args.no_plots:
        overrides["plots"] = False
    if args.resume:
        overrides["resume"] = args.resume
    CSPNDepthTrainer(overrides=overrides, cspn_steps=args.cspn_steps, norm_type=args.norm_type,
                     n_sample=args.n_sample, train_scope=args.train_scope).train()


def predict(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)  # resolves CSPNDepth* classes because this script defines them
    results = model(args.source, imgsz=args.imgsz, device=args.device)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm

        for i, r in enumerate(results):
            depth = r.depth.data.float().cpu().numpy().squeeze()
            plt.imsave(out_dir / f"depth_{i}.png", depth, cmap=cm.turbo)
            print(f"saved {out_dir / f'depth_{i}.png'}  shape={depth.shape} "
                  f"range=[{depth.min():.2f}, {depth.max():.2f}] m")
    except ImportError:
        torch.save([r.depth.data.cpu() for r in results], out_dir / "depths.pt")
        print(f"matplotlib missing; raw depths saved to {out_dir / 'depths.pt'}")


def self_test(args):
    """CPU smoke test: build the wrapped model, run train/eval forwards, check gradient flow."""
    torch.manual_seed(0)
    model = CSPNDepthModel("yolo26n-depth.yaml", cspn_steps=6, n_sample=100)
    model.args = DEFAULT_CFG
    x = torch.randn(1, 3, 256, 320)
    gt = torch.rand(1, 256, 320) * 5 + 0.5  # meters

    model.train()
    loss, _ = model.loss({"img": x, "depth": gt})  # trainer contract: loss is a vector, .sum() it
    loss = loss.sum()
    loss.backward()
    g = model.model[-1].guide[0].conv.weight.grad
    inner_g = model.model[-1].inner.head[-1].weight.grad
    assert g is not None and torch.isfinite(g).all(), "no/NaN grad into guidance branch"
    print(f"train fwd OK  loss={loss.item():.4f}  guide.grad={'finite' if g is not None else None}  "
          f"inner_head.grad={'finite' if inner_g is not None and torch.isfinite(inner_g).all() else 'none'}")

    model.eval()
    with torch.no_grad():
        pred = model(x)
    assert pred.shape[:2] == (1, 1) and torch.isfinite(pred).all()
    print(f"eval  fwd OK  pred={tuple(pred.shape)}  range=[{pred.min():.3f}, {pred.max():.3f}] m")
    print("self-test passed ✔")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predict", action="store_true", help="run inference instead of training")
    p.add_argument("--self-test", action="store_true", help="CPU smoke test of the wrapped model")
    p.add_argument("--model", default="yolo26n-depth.pt", help="base weights for training")
    p.add_argument("--data", default="nyu-depth.yaml", help="depth dataset yaml (images/ + depth/)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr0", type=float, default=1e-4)
    p.add_argument("--name", default="cspn_depth")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--no-plots", action="store_true", help="disable plots (useful offline)")
    p.add_argument("--resume", default=None, help="path to last.pt to resume")
    # CSPN options
    p.add_argument("--cspn-steps", type=int, default=24, help="CSPN propagation steps")
    p.add_argument("--norm-type", default="8sum", choices=["8sum", "8sum_abs"])
    p.add_argument("--n-sample", type=int, default=500,
                   help="sparse points sampled from GT each train step; 0 = pure refinement")
    p.add_argument("--train-scope", default="guide", choices=["guide", "head", "all"])
    # predict options
    p.add_argument("--weights", default=None, help="checkpoint for --predict")
    p.add_argument("--source", default=None, help="image/dir for --predict")
    p.add_argument("--save-dir", default="predict_out")
    args = p.parse_args()

    if args.self_test:
        self_test(args)
    elif args.predict:
        assert args.weights and args.source, "--predict needs --weights and --source"
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
