# New Model: Causal FIND

This directory keeps only the causal FIND model used as the project new model.

## Files

- `causal.py`: causal FIND model, training loop, checkpoint loading, and TPR@target-FPR utility.
- `utils.py`: MNW evaluation, checkpoint helpers, ROC plotting, and Hugging Face token helpers.
- `config.py`: default paths and hyperparameters.
- `main.py`: command-line entry point for `train-causal` and `test-causal`.

Shared dataloading code lives in `data/find_dataset.py` at the project level.

## Data

Run commands from inside `Code/`. Default paths are:

```text
data/tiny-genimage/
  train/
  val/

data/MNW/AI_Images/
```

The dataloader scans `train` and `val`, treats folders named `real`, `nature`, `original`, or `0_real` as real images, and treats other generator folders as fake images.

The dataloader first tries to read `data/manifests/train.csv` and `data/manifests/val.csv`. If those files are empty, it falls back to directory scanning. MNW testing first tries `data/manifests/test.csv`, then falls back to `data/MNW/AI_Images`.

## Train

```bash
python -m new_model.main train-causal \
  --data-dir data/tiny-genimage \
  --output-dir runs/causal
```

Device selection defaults to CUDA when available, then MPS on macOS, then CPU. You can override it with `--device cuda`, `--device mps`, or `--device cpu`.

Common options:

```bash
python -m new_model.main train-causal \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-4 \
  --causal-mode full \
  --ood-generator-fraction 0.25
```

## MNW Test

MNW evaluation uses AI images from `data/MNW/AI_Images` and real images from Hugging Face `ILSVRC/imagenet-1k` validation streaming.

Set a Hugging Face token first:

```bash
export HF_TOKEN="your_huggingface_token"
```

Then run:

```bash
python -m new_model.main test-causal --output-dir runs/causal
```

Useful evaluation options:

```bash
python -m new_model.main test-causal \
  --output-dir runs/causal \
  --threshold-from-target-fpr 0.05 \
  --roc-target-fpr 0.05
```

Outputs include:

- `test_results.npz`
- `evaluation_reports.json`
- `find_roc.png`
