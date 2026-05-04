# DDA4210 Course Project Code

This is the only folder intended for submission and GitHub upload.

## Structure

```text
Code/
  baseline/       # unified baseline: ResNet18 and frozen CLIP
  new_model/      # causal FIND model
  data/           # dataset download/preparation scripts and manifests
  README.md
  requirements.txt
```

The project uses Tiny-GenImage for training/validation and MNW for final testing.

## Data Layout

Place datasets under this folder:

```text
Code/
  data/
    tiny-genimage/
      train/
      val/
    MNW/
      AI_Images/
    manifests/
      train.csv
      val.csv
      test.csv
```

Label convention:

```text
real = 0
fake = 1
```

Build CSV manifests after placing or downloading data:

```bash
python data/scripts/build_manifests.py \
  --tiny-genimage-root data/tiny-genimage \
  --mnw-root data/MNW/AI_Images \
  --output-dir data/manifests
```

Training and testing code read from `data/manifests/*.csv` first. The committed `test.csv` is intentionally empty except for its header.

Check data readiness with:

```bash
python data/scripts/check_data.py
```

## Install

```bash
cd Code
pip install -r requirements.txt
```

## Baseline

Run from inside `Code/`:

```bash
python baseline/train.py --model resnet18
python baseline/train.py --model clip
```

Device selection defaults to CUDA when available, then MPS on macOS, then CPU. You can override it with `--device cuda`, `--device mps`, or `--device cpu`.

MNW evaluation:

```bash
export HF_TOKEN="your_huggingface_token"

python baseline/evaluate_mnw.py \
  --checkpoint baseline/outputs/resnet18_seed4210/best.pt
```

See `baseline/README.md` for more options.

## New Model: Causal FIND

Run from inside `Code/`:

```bash
python -m new_model.main train-causal \
  --data-dir data/tiny-genimage \
  --output-dir runs/causal
```

MNW evaluation:

```bash
python -m new_model.main test-causal --output-dir runs/causal
```

See `new_model/README.md` for causal-model options and threshold settings.
