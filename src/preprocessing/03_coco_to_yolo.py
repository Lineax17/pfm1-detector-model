from pathlib import Path
import json
import shutil

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
YOLO_ROOT = REPO_ROOT / "data" / "yolo"


def coco_to_yolo_split(processed_root: Path, output_root: Path, split: str):
    """
    Convert one split (train/val) from COCO JSON to YOLO format
    """

    images_dir = processed_root / split / "images"
    annotations_file = processed_root / split / "annotations.json"

    out_images = output_root / split / "images"
    out_labels = output_root / split / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # --- load json ---
    with open(annotations_file) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}

    # category mapping → 0-based
    categories = sorted(data["categories"], key=lambda x: x["id"])
    cat_id_map = {cat["id"]: i for i, cat in enumerate(categories)}

    # --- group annotations by image ---
    anns_per_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        anns_per_image.setdefault(img_id, []).append(ann)

    # --- process images ---
    for img_id, img_info in images.items():
        file_name = Path(img_info["file_name"]).name
        src_img_path = images_dir / file_name

        if not src_img_path.exists():
            print(f"⚠️ Missing image: {src_img_path}")
            continue

        # copy image
        dst_img_path = out_images / file_name
        shutil.copy2(src_img_path, dst_img_path)

        # create label file
        label_path = out_labels / (file_name.replace(Path(file_name).suffix, ".txt"))

        width = img_info["width"]
        height = img_info["height"]

        anns = anns_per_image.get(img_id, [])

        lines = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # convert to YOLO format (normalized)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            class_id = cat_id_map[ann["category_id"]]

            lines.append(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}")

        # write file (auch leer wenn keine bbox → wichtig!)
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Finished {split}")


def convert_dataset():
    for split in ["train", "val", "test"]:
        coco_to_yolo_split(PROCESSED_ROOT, YOLO_ROOT, split)

    print("Conversion complete!")


if __name__ == "__main__":
    convert_dataset()