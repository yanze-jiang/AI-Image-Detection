# HybridForensics Benchmark

这个目录用于存放第二 benchmark：`HybridForensics / Real and Fake AI-Generated Standardized 512px Image Forensics Dataset`。

## 当前归档位置

原始下载目录已经整理到：

```text
data/benchmarks/hybridforensics/raw
```

## 数据结构

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
  Readme.md
```

## 数据规模

- `Real/FFHQ`: `2500`
- `Real/MS_COCO`: `2500`
- `Fake_GAN/ProGAN`: `1250`
- `Fake_GAN/StyleGAN3`: `1250`
- `Fake_Diffusion/Midjourney`: `1250`
- `Fake_Diffusion/SDXL`: `1250`

总计：

- `5000 real`
- `5000 fake`

## 用途建议

这个 benchmark 更适合作为：

1. 外部测试集
2. 补充泛化评估
3. `CIFAKE` 之外的第二跑分 benchmark

不建议一开始直接拿它作为主训练集。
