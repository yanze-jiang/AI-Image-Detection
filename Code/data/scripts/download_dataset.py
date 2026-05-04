from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Tiny-GenImage-style data from Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset or repository id.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/tiny-genimage"))
    parser.add_argument("--repo-type", choices=["dataset", "model", "space"], default="dataset")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(args.output_dir),
        token=args.token,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.repo_id} to {args.output_dir}")


if __name__ == "__main__":
    main()
