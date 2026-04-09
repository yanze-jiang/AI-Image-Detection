# HybridForensics 外部 Benchmark 记录

## 1. 数据集定位

为了避免只在 `CIFAKE` 单一分布上评估模型，本项目额外引入 `HybridForensics` 作为第二 benchmark，用于补充跨数据集泛化测试。

该 benchmark 的特点是：

1. 数据规模适中：共 `10000` 张图像
2. 二分类结构清晰：`5000 real + 5000 fake`
3. 假图来源更复杂：同时包含 `GAN` 与 `Diffusion` 两类生成范式
4. 更接近外部分布测试，而不是单一训练集内测试

## 2. 当前数据结构

归档位置：

```text
data/benchmarks/hybridforensics/raw
```

原始结构：

```text
raw/
  Real/
    FFHQ/
    MS_COCO/
  Fake_GAN/
    ProGAN/
    StyleGAN3/
  Fake_Diffusion/
    Midjourney/
    SDXL/
```

## 3. 首轮测试设置

当前已经完成一次外部 benchmark 测试，所使用的模型为：

`MobileNetV3-Small baseline`

checkpoint 路径：

```text
baseline/outputs/mobilenet_v3_small/one_epoch_test/best.pt
```

测试脚本：

```text
baseline/evaluate_hybridforensics.py
```

## 4. 首轮结果

有效样本数：`9999`

有 `1` 张图像损坏，被评估脚本自动跳过：

```text
data/benchmarks/hybridforensics/raw/Real/MS_COCO/MS_COCO_1354.jpg
```

当前结果如下：

1. `Accuracy = 0.5231`
2. `AUC = 0.5919`
3. `F1 = 0.6672`
4. `Loss = 1.7824`

## 5. 结果解释

这个结果说明：

1. 模型在 `CIFAKE` 上训练后，迁移到 `HybridForensics` 时表现明显下降
2. `AUC` 仅为 `0.59` 左右，说明模型只学到有限的可迁移判别信息
3. 仅在单一数据集上取得高分，并不意味着具有良好的跨数据集泛化能力

因此，这个 benchmark 的意义非常明确：

**它为后续双流融合模型提供了一个更有挑战性的外部测试场景。**

## 6. 可用于报告/PPT 的表述

可以直接使用下面这段：

> 为了验证模型是否真正学习到稳定的伪造特征，而不是仅适应 CIFAKE 的数据分布，我们引入 HybridForensics 作为外部 benchmark。初步实验表明，轻量 CNN baseline 在 CIFAKE 上可取得较高性能，但迁移到 HybridForensics 后 AUC 仅为 0.5919，说明其跨数据集泛化能力有限。这为后续引入取证分支和双流融合提供了明确动机。

## 7. 后续建议

接下来建议按相同方式继续测试：

1. `ResNet18`
2. `CLIP + linear head`
3. `Dual-stream`

这样就可以形成一组完整的外部 benchmark 对比结果。
