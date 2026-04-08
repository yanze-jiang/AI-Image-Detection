# 数据目录说明

本目录用于执行 `Idea/data_protocol.md` 中定义的数据准备流程。

## 目录结构

```text
data/
  raw/
    cifake/
      train/
        REAL/
        FAKE/
      test/
        REAL/
        FAKE/
  processed/
  perturbed/
  configs/
    cifake.json
```

## 你现在需要做的事

1. 把解压后的 `CIFAKE` 数据放入 `data/raw/cifake/`。
2. 运行 `scripts/prepare_cifake.py` 生成 `processed/` 数据。
3. 运行 `scripts/build_perturbations.py` 为测试集生成压缩和缩放扰动版本。
4. 在 `results/results_template.csv` 中记录实验结果。

## 原始数据放置规则

建议按下面方式放图：

- `data/raw/cifake/train/REAL/`
- `data/raw/cifake/train/FAKE/`
- `data/raw/cifake/test/REAL/`
- `data/raw/cifake/test/FAKE/`

脚本会递归扫描这些目录下的常见图片格式文件。

## 生成后的目录结构

数据预处理脚本会生成：

```text
data/processed/
  train/
    real/*.jpg
    fake/*.jpg
  val/
    real/*.jpg
    fake/*.jpg
  test/
    real/*.jpg
    fake/*.jpg
  manifests/
    train.csv
    val.csv
    test.csv
  split_summary.json
```

扰动脚本会生成：

```text
data/perturbed/
  test_jpeg95/
  test_jpeg85/
  test_jpeg75/
  test_resize/
```

## 运行命令

先准备子集并统一预处理：

```bash
python "scripts/prepare_cifake.py" --config "data/configs/cifake.json"
```

再生成测试扰动：

```bash
python "scripts/build_perturbations.py" --source-root "data/processed" --output-root "data/perturbed"
```

## 依赖

脚本依赖 `Pillow`。如果本地没有，可安装：

```bash
pip install pillow
```

如果你还没有下载 `CIFAKE`，可以从 Kaggle 下载并解压后放进 `data/raw/cifake/`。

如果你使用 Kaggle API，常见流程是：

```bash
pip install kaggle
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p "data/raw"
```

下载后把压缩包解压，并整理成上面的 `train/REAL`、`train/FAKE`、`test/REAL`、`test/FAKE` 结构即可。
