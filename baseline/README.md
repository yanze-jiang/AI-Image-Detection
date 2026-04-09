# Baseline 工程说明

本目录包含三个 baseline：

1. `ResNet18`  
   作为传统 CNN baseline，对应经典残差网络论文：
   `He et al., Deep Residual Learning for Image Recognition, CVPR 2016`

2. `MobileNetV3-Small`  
   作为更轻量、更弱的 CNN baseline，适合作为和 `ResNet18` 对照的低容量模型：
   `Howard et al., Searching for MobileNetV3, ICCV 2019`

3. `CLIP + linear head`  
   作为预训练视觉表征 baseline，对应 CLIP 论文：
   `Radford et al., Learning Transferable Visual Models From Natural Language Supervision, ICML 2021`

## 目录说明

- `common.py`：数据加载、模型构建、训练和评估的公共函数
- `train_resnet18.py`：训练 `ResNet18 baseline`
- `train_mobilenetv3_small.py`：训练 `MobileNetV3-Small baseline`
- `train_clip.py`：训练冻结 `CLIP` 编码器加线性分类头
- `evaluate.py`：评估某个 checkpoint 在指定测试集上的表现
- `record_result.py`：将结果追加写入 `results/results_template.csv`

## 数据前提

默认使用下面目录：

```text
data/processed/
  train/
  val/
  test/

data/perturbed/
  test_jpeg95/
  test_jpeg85/
  test_jpeg75/
  test_resize/
```

## 1. 训练 ResNet18 baseline

```bash
python "baseline/train_resnet18.py"
```

如果要做小样本对比：

```bash
python "baseline/train_resnet18.py" --small-train-count 10000
```

## 2. 训练 CLIP baseline

先训练更弱 CNN baseline：

```bash
python "baseline/train_mobilenetv3_small.py"
```

如果要做小样本对比：

```bash
python "baseline/train_mobilenetv3_small.py" --small-train-count 10000
```

## 3. 训练 CLIP baseline

```bash
python "baseline/train_clip.py"
```

如果要做小样本对比：

```bash
python "baseline/train_clip.py" --small-train-count 10000
```

第一次运行 `CLIP` 时，`transformers` 会自动下载 `openai/clip-vit-base-patch32` 权重。

## 4. 评估 checkpoint

评估标准测试集：

```bash
python "baseline/evaluate.py" \
  --checkpoint "baseline/outputs/resnet18/best.pt" \
  --data-root "data/processed" \
  --split "test"
```

评估扰动测试集：

```bash
python "baseline/evaluate.py" \
  --checkpoint "baseline/outputs/resnet18/best.pt" \
  --data-root "data/perturbed" \
  --split "test_jpeg85"
```

## 5. 记录结果

```bash
python "baseline/record_result.py" \
  --csv-path "results/results_template.csv" \
  --model-name "resnet18" \
  --train-size "full" \
  --test-setting "test" \
  --perturbation "none" \
  --accuracy 0.9500 \
  --auc 0.9800 \
  --f1 0.9490
```

## 建议实验顺序

1. 先跑 `MobileNetV3-Small full`
2. 再跑 `MobileNetV3-Small small`
3. 再跑 `ResNet18 full`
4. 再跑 `ResNet18 small`
5. 再跑 `CLIP full`
6. 再跑 `CLIP small`
7. 对最好的 baseline 跑 `test_jpeg95 / test_jpeg85 / test_jpeg75 / test_resize`
