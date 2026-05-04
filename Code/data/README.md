# Data Interface

This folder contains only data preparation code and manifest files. Raw images and downloaded datasets are ignored by Git.

## Manifest Format

All training and evaluation code can read CSV manifests with this schema:

```csv
path,label,split,generator
```

- `path`: image path, relative to `Code/` or absolute.
- `label`: `real` / `fake` or `0` / `1`.
- `split`: `train`, `val`, or `test`.
- `generator`: generator name. Use `real` for real images.

The committed manifest files are intentionally empty except for headers:

```text
data/manifests/train.csv
data/manifests/val.csv
data/manifests/test.csv
```

After downloading or placing data locally, run:

```bash
python data/scripts/build_manifests.py \
  --tiny-genimage-root data/tiny-genimage \
  --mnw-root data/MNW/AI_Images \
  --output-dir data/manifests
```

## Download

Use `download_dataset.py` when a Hugging Face dataset/repository id is available:

```bash
python data/scripts/download_dataset.py \
  --repo-id your-org-or-user/tiny-genimage \
  --output-dir data/tiny-genimage
```

If the course provides data as an archive, unpack it into `data/tiny-genimage/` and then build manifests.

## Check

Verify that folders and manifests are in place:

```bash
python data/scripts/check_data.py
```
