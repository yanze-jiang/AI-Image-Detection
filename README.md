# DDA4210 Course Project

本仓库用于完成课程项目 **“面向资源受限场景的 AI 生成图像检测”**。项目围绕 `CIFAKE` 主训练集、`Hemg` 自划分补充训练集、压缩/缩放扰动测试，以及 `HybridForensics` 外部 benchmark 展开，目标是在有限算力条件下构建一个 **轻量、可复现、便于对比** 的图像真伪检测方案。

当前工程已经包含：

- 数据准备与扰动构建脚本
- `Hemg` 本地切分脚本
- `MobileNetV3-Small`、`ResNet18`、`CLIP + linear head` baseline
- `HybridForensics` 外部测试脚本
- 项目方案、实验计划与汇报材料草稿

统一复现实验时，默认随机种子固定为 `4210`。

## 当前项目状态

目前已经完成首轮 baseline 与外部 benchmark 验证。初步结果表明：模型在原训练分布上可以取得较高分数，但迁移到 `HybridForensics` 时性能明显下降，这说明跨数据集泛化仍然是本项目的核心问题，也构成后续双流融合模型的主要动机。

后续改进方向为：

- 语义流：预训练视觉编码器特征
- 取证流：`SRM` 残差或 `DCT` 频域特征
- 融合方式：特征拼接 + 小型 `MLP` 分类头

## 仓库结构

- `idea/`：项目方案、数据协议、baseline 计划、融合模型计划、汇报材料建议
- `scripts/`：数据准备、子集构建、扰动生成脚本
- `baseline/`：baseline 训练、评估、结果记录与 `HybridForensics` 测试代码
- `data/CIFAKE/`：`CIFAKE` 数据处理说明与配置
- `data/hemg_processed/`：`Hemg` 切分后的训练目录与说明
- `data/hybridforensics/`：外部 benchmark 说明
- `results/`：实验结果记录模板

## 环境依赖

建议使用 `Python 3.10+`。

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 准备 `CIFAKE`

```bash
python "scripts/prepare_cifake.py" --config "data/CIFAKE/cifake.json"
python "scripts/build_perturbations.py" --source-root "data/processed" --output-root "data/perturbed"
```

### 2. 准备 `Hemg`

```bash
python "scripts/download_hemg.py"
python "scripts/prepare_hemg_split.py" --clean
```

### 3. 训练 baseline

```bash
python "baseline/train_mobilenetv3_small.py" --seed 4210
python "baseline/train_resnet18.py" --seed 4210
python "baseline/train_clip.py" --seed 4210
```

如果使用 `Hemg`，把 `--data-root` 改成 `data/hemg_processed` 即可。

### 4. 评估标准测试集

```bash
python "baseline/evaluate.py" \
  --checkpoint "baseline/outputs/resnet18/best.pt" \
  --data-root "data/processed" \
  --split "test"
```

### 5. 评估 `HybridForensics`

```bash
python "baseline/evaluate_hybridforensics.py" \
  --checkpoint "baseline/outputs/mobilenet_v3_small/one_epoch_test/best.pt"
```

如需查看 baseline 训练与评估细节，请阅读 `baseline/README.md`。

## GitHub 协作约定

为了保证仓库适合组内协作，本仓库默认 **不提交** 大体积数据、模型权重和训练输出。以下内容应保留在本地或通过共享网盘分发：

- `data/raw/`
- `data/processed/`
- `data/perturbed/`
- `baseline/outputs/`
- `cifake.zip`

建议组内统一采用下面的协作流程：

1. 每位成员从 `main` 拉取最新代码。
2. 为自己的任务创建新分支，例如 `feat-dual-stream`、`docs-report-update`。
3. 在个人分支提交并推送改动。
4. 通过 GitHub `Pull Request` 合并到 `main`。

常用命令：

```bash
git checkout -b feat-your-task
git add .
git commit -m "describe your change"
git push -u origin feat-your-task
```

## 进一步阅读

- 项目说明：`idea/README.md`
- 方案初稿：`idea/ver1.md`
- `CIFAKE` 数据准备：`data/CIFAKE/README.md`
- `Hemg` 数据准备：`data/hemg_processed/README.md`
- baseline 说明：`baseline/README.md`
