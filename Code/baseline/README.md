# Baseline

整理版 baseline，只保留作业需要的统一训练和 MNW 测试入口。

## 统一约定

- 训练/验证数据集：Tiny-GenImage
- 最终测试集：MNW
- 标签定义：`real = 0`，`fake = 1`
- 输入尺寸：`224 x 224`
- 默认随机种子：`4210`

## 数据目录

从 `Code/` 目录运行时，训练脚本默认读取：

```text
data/tiny-genimage/
  train/
  val/
```

训练会优先读取：

```text
data/manifests/train.csv
data/manifests/val.csv
```

如果 manifest 为空或不存在，才回退到扫描 `data/tiny-genimage/` 目录。

也可以显式指定：

```bash
python baseline/train.py \
  --model resnet18 \
  --data-root /path/to/tiny-genimage
```

MNW 测试默认读取：

```text
data/MNW/AI_Images/
```

MNW 假图会优先从 `data/manifests/test.csv` 读取；该文件当前只保留表头，放入测试集后运行 `data/scripts/build_manifests.py` 即可生成内容。

真实图默认使用 Hugging Face 的 `ILSVRC/imagenet-1k` validation streaming；如果已有本地真实图，可以通过 `--real-dir` 指定。

## 训练

```bash
python baseline/train.py --model resnet18
python baseline/train.py --model mobilenet_v3_small
python baseline/train.py --model clip
```

默认设备优先级为 CUDA、MPS、CPU。可用 `--device cuda`、`--device mps` 或 `--device cpu` 指定。

输出默认保存到：

```text
baseline/outputs/
```

## MNW 测试

```bash
export HF_TOKEN="your_huggingface_token"

python baseline/evaluate_mnw.py \
  --checkpoint baseline/outputs/resnet18_seed4210/best.pt \
  --mnw-dir data/MNW/AI_Images \
  --save-json baseline/outputs/resnet18_seed4210/mnw_eval.json
```

使用本地真实图：

```bash
python baseline/evaluate_mnw.py \
  --checkpoint baseline/outputs/resnet18_seed4210/best.pt \
  --mnw-dir data/MNW/AI_Images \
  --real-dir /path/to/real_images
```
