"""Convert COCO-format annotations to YOLO-format label files."""

from pathlib import Path
import json
import shutil

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
YOLO_ROOT = REPO_ROOT / "data" / "yolo"


def coco_to_yolo_split(
    processed_root: Path, output_root: Path, split: str
) -> None:
    """Convert one dataset split from COCO JSON to YOLO label format.

    Each annotation is written as a ``.txt`` file alongside the copied image,
    with one line per bounding box::

        <class_id> <x_center> <y_center> <width> <height>

    where all coordinate values are normalised to [0, 1].

    Parameters
    ----------
    processed_root : Path
        Root of the processed COCO dataset.
    output_root : Path
        Root where the YOLO-format dataset will be written.
    split : str
        Split name (``"train"``, ``"val"``, or ``"test"``).
    """
    images_dir = processed_root / split / "images"
    annotations_file = processed_root / split / "annotations.json"

    out_images = output_root / split / "images"
    out_labels = output_root / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Load COCO JSON
    with open(annotations_file) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}

    # Map category IDs to zero-based indices
    categories = sorted(data["categories"], key=lambda x: x["id"])
    cat_id_map = {cat["id"]: i for i, cat in enumerate(categories)}

    # Group annotations by image ID
    anns_per_image: dict[int, list] = {}
    for ann in data["annotations"]:
        anns_per_image.setdefault(ann["image_id"], []).append(ann)

    # Process each image
    for img_id, img_info in images.items():
        file_name = Path(img_info["file_name"]).name
        src_img_path = images_dir / file_name

        if not src_img_path.exists():
            print(f"Missing image: {src_img_path}")
            continue

        # Copy image
        dst_img_path = out_images / file_name
        shutil.copy2(src_img_path, dst_img_path)

        # Write YOLO label file (may be empty if no annotations)
        label_path = out_labels / (
            file_name.replace(Path(file_name).suffix, ".txt")
        )
        width = img_info["width"]
        height = img_info["height"]

        lines = []
        for ann in anns_per_image.get(img_id, []):
            x, y, w, h = ann["bbox"]

            # Convert COCO [x, y, w, h] to YOLO [x_center, y_center, w, h] normalised
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            class_id = cat_id_map[ann["category_id"]]
            lines.append(
                f"{class_id} {x_center} {y_center} {w_norm} {h_norm}"
            )

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Finished {split}")


def convert_dataset():
    """Convert all splits (train / val / test) to YOLO format."""
    for split in ["train", "val", "test"]:
        coco_to_yolo_split(PROCESSED_ROOT, YOLO_ROOT, split)
    print("Conversion complete!")


if __name__ == "__main__":
    convert_dataset()
