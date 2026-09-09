# Depth

CSPN 精化 × YOLO26-Depth 单目深度估计：在 Ultralytics 官方 depth 模型的解码头后接 CSPN
（Convolutional Spatial Propagation Network）传播精化，两个三方仓库（`ultralytics/`、`CSPN/`）零改动，
集成代码全部在 `train_yolo_cspn.py`。

```text
RGB ─► YOLO26 backbone/FPN ─► Depth 解码头（预训练层复用）
                                ├─ 粗深度   (B,1,H/4,W/4)   米制，exp 头
                                └─ guidance (B,8,H/4,W/4)   新增分支，从融合特征生成
                                          └─► CSPN（零参数，24 步传播）─► 精化深度
```

- loss / 验证 / checkpoint 全部走 Ultralytics 原生流水线，监督对象是 **CSPN 精化后的输出**。
- 训练时每步从 GT 随机采样 `--n-sample` 个稀疏点，在 CSPN 传播中钉住测量值（深度补全模式）；
  `--n-sample 0` 则为纯单目精化模式。
- 参数量：总 5.34M，其中 guidance 分支 0.169M，整个解码头 3.08M，backbone+neck 2.26M。

## 环境准备（一次）

```bash
conda activate cspn          # 本机已建好：Python 3.11 + torch 2.11.0+cu128（RTX 5090）
```

- Git Bash 里若 SSL 报错，先执行：
  `export PATH="/d/miniconda3/envs/cspn:/d/miniconda3/Library/bin:$PATH"`
- **官方预训练权重**：GitHub 被墙，需手动（浏览器/代理）下载
  `yolo26n-depth.pt`（[下载地址](https://github.com/ultralytics/assets/releases)）放到 `D:\Code\Depth\`。
- 冒烟测试：`python train_yolo_cspn.py --self-test`
- 合成数据全链路测试（可选，数据已生成在 `testdata/`）：

```bash
python train_yolo_cspn.py --model yolo26n-depth.yaml --data testdata/fake-depth.yaml \
    --epochs 1 --imgsz 256 --batch 4 --device 0 --workers 2 \
    --train-scope guide --cspn-steps 6 --n-sample 100 --no-plots --name e2e_test --exist-ok
```

## 数据集

`images/<split>/*.jpg` 与 `depth/<split>/*.png` 同名配对；深度图为 uint16 毫米 PNG
（或 float32 米制 `.npy`），`depth_scale` 在 yaml 里指定。yaml 模板：

```yaml
path: D:/Code/Depth/testdata   # 数据根目录（建议绝对路径）
train: images/train
val: images/val
nc: 1
names:
  0: depth
channels: 3
depth_scale: 1000              # PNG 值乘以 1/1000 = 米
```

官方 NYU Depth V2：直接 `--data nyu-depth.yaml` 首次自动下载（约 1.5G，走 GitHub 可能需要代理）。

## 两阶段训练

| | 阶段一（guide） | 阶段二（all） |
|---|---|---|
| 训练内容 | 只训 guidance 分支（0.169M） | 全网联合微调（5.34M） |
| 学习率 | 1e-4（正常） | 1e-5（小，防灾难性遗忘） |
| 目的 | 让亲和度先学会"怎么传"，稳定、便宜、不碰预训练权重 | 让粗深度也适应 CSPN 的存在，两个模块磨合 |
| 风险 | 无 | 微调会动预训练主干，数据少时泛化会退化 |

直接端到端训（跳过阶段一）在小数据上容易把预训练权重带偏；先冻结练 guide，
阶段一结束就有一个可用的模型，阶段二只是锦上添花。

### 阶段一：只训 guidance 分支

```bash
python train_yolo_cspn.py \
    --model yolo26n-depth.pt \
    --data nyu-depth.yaml \
    --epochs 5 \
    --imgsz 640 --batch 16 --device 0 \
    --train-scope guide \
    --cspn-steps 24 --n-sample 500 \
    --lr0 1e-4 \
    --name stage1
```

- 冻结范围：backbone、neck、解码头全部冻结（含 BN 统计量），只训 guide 分支。
- 产物：`ultralytics/runs/depth/stage1/weights/best.pt`
- 显存不够就先减 `--batch` 或 `--cspn-steps 12`（正式推理时仍可用 24 步）。

### 阶段二：联合微调

```bash
python train_yolo_cspn.py \
    --model ultralytics/runs/depth/stage1/weights/best.pt \
    --data nyu-depth.yaml \
    --epochs 20 \
    --imgsz 640 --batch 16 --device 0 \
    --train-scope all \
    --cspn-steps 24 --n-sample 500 \
    --lr0 1e-5 \
    --name stage2
```

- `--model` 直接指向阶段一的 best.pt，guidance 权重会被完整继承。
- 防遗忘两件事：小学习率（1e-5）；训练数据里混 5–10% 多样化图像
  （Ultralytics 官方对 depth 微调的同样建议）。
- 训完模型输出的"米"会漂，框架会在验证集上自动重校准（`Calibrating depth output scale`）。

## 冻结机制说明（--train-scope 三档）

冻结不是简单的 `requires_grad=False`——Ultralytics 的 trainer 会把**不在它冻结名单里的
已冻结参数强制解冻**（`BaseTrainer._setup_train`），所以本脚本通过官方的 `args.freeze`
名字匹配机制表达冻结范围，顺带白拿 BN 统计量冻结（每 epoch 自动把冻结层的 BN 设回 eval）：

| train-scope | 冻结 | 可训练 | 用途 |
|---|---|---|---|
| `guide` | backbone+neck+解码头（BN 一并冻结） | guidance 分支 0.169M | 阶段一 |
| `head` | backbone+neck | 整个解码头 3.08M | 中间档（想微调解码头但保主干） |
| `all` | 仅 CSPN 的固定全 1 求和卷积 | 全网 5.34M | 阶段二 |

## 推理与评估

```bash
# 推理单图/目录，输出 turbo 伪彩深度图
python train_yolo_cspn.py --predict \
    --weights ultralytics/runs/depth/stage2/weights/best.pt \
    --source path/to/img_or_dir --device 0 --save-dir predict_out
```

注意：checkpoint 里打包了本脚本定义的类，**推理必须经由本脚本**（或先
`import train_yolo_cspn` 再用 YOLO API 加载），换其他脚本直接 `YOLO("best.pt")` 会反序列化失败。

## 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `yolo26n-depth.pt` | 起点：官方 .pt（推荐）/ 自己的 checkpoint / `.yaml`（从零） |
| `--data` | `nyu-depth.yaml` | 数据集 yaml |
| `--train-scope` | `guide` | 冻结档位：`guide` / `head` / `all` |
| `--cspn-steps` | 24 | 传播步数；H/4 分辨率下 24 步≈全图 96px 传播半径 |
| `--norm-type` | `8sum` | `8sum`（允许负亲和度）/ `8sum_abs`（强制正） |
| `--n-sample` | 500 | 每步从 GT 采样的稀疏点数；0=纯精化模式 |
| `--lr0` | 1e-4 | 阶段一 1e-4，阶段二建议 1e-5 |
| `--no-plots` | off | 离线机器禁用绘图（避免字体下载失败） |

## 已知实现细节（踩过的坑）

- **CSPN 传播内部强制 fp32**：亲和度归一化的除法在 AMP fp16 下数值下溢产生 NaN 梯度，
  因此 `CSPN.forward` 里用 `autocast(enabled=False)` 建了 fp32 计算岛；EMA/验证时模型被
  `.half()`，固定全 1 求和卷积通过 `_apply` 钩子始终保持 fp32。
- **channels_last 已禁用**：与 CSPN 的 5 维求和卷积不兼容，5M 小模型无速度损失。
- **`batch["depth"]` 形状**：trainer 里是 (B,1,H,W)，稀疏采样已做兼容。
- 官方权重下载走 GitHub，本机网络需代理或手动下载。
