# DDA4210 Course Project

本仓库用于组内协作完成 AI 图像检测课程项目，当前内容包括项目方案、数据准备脚本、baseline 训练与评估代码，以及结果记录模板。

## 仓库内容

- `Idea/`：项目方案、数据协议、baseline 计划、融合模型计划、汇报材料建议
- `scripts/`：数据集整理与扰动构建脚本
- `baseline/`：`MobileNetV3-Small`、`ResNet18` 与 `CLIP + linear head` baseline
- `data/configs/`：数据准备配置
- `results/`：结果记录模板

## 协作约定

为了保证仓库可直接在 GitHub 协作，本仓库默认不提交大体积数据、模型权重和训练输出，以下内容会保留在本地：

- `data/raw/`
- `data/processed/`
- `data/perturbed/`
- `baseline/outputs/`
- `cifake.zip`

如果组员需要相同数据，请按 `data/README.md` 中的流程自行下载和预处理，或通过共享网盘统一分发。

## 环境依赖

建议使用 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

1. 准备数据

```bash
python "scripts/prepare_cifake.py" --config "data/configs/cifake.json"
python "scripts/build_perturbations.py" --source-root "data/processed" --output-root "data/perturbed"
```

2. 训练 baseline

```bash
python "baseline/train_mobilenetv3_small.py"
python "baseline/train_resnet18.py"
python "baseline/train_clip.py"
```

3. 评估模型

```bash
python "baseline/evaluate.py" \
  --checkpoint "baseline/outputs/resnet18/best.pt" \
  --data-root "data/processed" \
  --split "test"
```

## 进一步阅读

- 项目说明：`Idea/README.md`
- 数据准备：`data/README.md`
- baseline 说明：`baseline/README.md`
