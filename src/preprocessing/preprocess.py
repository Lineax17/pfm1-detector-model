from __future__ import annotations

import argparse
from pathlib import Path

from helpers import Preprocessor

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "data" / "merged"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"


def preprocess_merged(
    merged_root: Path,
    output_root: Path,
    image_size: tuple[int, int],
    splits: tuple[str, ...],
) -> None:
    preprocessor = Preprocessor()

    for split in splits:
        split_images_dir = merged_root / split / "images"
        annotation_path = merged_root / split / "annotations.json"
        target_path = output_root / split

        if not split_images_dir.exists() or not annotation_path.exists():
            print(f"Skip '{split}': missing images or annotations.")
            continue

        print(f"Preprocess '{split}' -> {target_path}")
        preprocessor.preprocess_folder(
            folder_path=split_images_dir,
            target_path=target_path,
            annotation_path=annotation_path,
            image_size=image_size,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess merged COCO data into data/processed.",
    )

    parser.add_argument(
        "--merged-root",
        type=Path,
        default=INPUT_DIR,
        help="Path to merged dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_DIR,
        help="Path to write processed dataset.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(320, 320),
        metavar=("WIDTH", "HEIGHT"),
        help="Target image size (width height).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Splits to process.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    preprocess_merged(
        merged_root=args.merged_root,
        output_root=args.output_root,
        image_size=tuple(args.image_size),
        splits=tuple(args.splits),
    )


if __name__ == "__main__":
    main()
