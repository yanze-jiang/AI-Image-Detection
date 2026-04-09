from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "Hemg/AI-Generated-vs-Real-Images-Datasets"
TARGET_DIR = Path(
    "data/Hemg"
)


def main() -> None:
    if TARGET_DIR.exists() and any(TARGET_DIR.iterdir()):
        raise FileExistsError(
            f"Target directory is not empty: {TARGET_DIR}. "
            "Please clear it first if you want a fresh download."
        )

    dataset = load_dataset(DATASET_NAME)
    dataset.save_to_disk(str(TARGET_DIR))
    print(f"Saved dataset to: {TARGET_DIR}")


if __name__ == "__main__":
    main()