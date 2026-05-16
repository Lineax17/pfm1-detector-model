import json
from pathlib import Path

def fix_coco(input_json, output_json, prefix=None):
    """
    Fix COCO JSON für Ultralytics YOLO

    Args:
        input_json: Pfad zur originalen JSON
        output_json: Pfad zur gefixten JSON
        prefix: optionaler Prefix für file_name (z.B. "train/images/")
    """

    with open(input_json, "r") as f:
        data = json.load(f)

    # --- Fix images ---
    for img in data.get("images", []):
        if prefix:
            # Nur hinzufügen wenn nicht schon drin
            if not img["file_name"].startswith(prefix):
                img["file_name"] = f"{prefix}{img['file_name']}"

    # --- Fix annotations ---
    fixed_annotations = []
    for i, ann in enumerate(data.get("annotations", [])):
        # ID hinzufügen falls fehlt
        ann["id"] = ann.get("id", i + 1)

        # area berechnen falls fehlt
        if "area" not in ann:
            x, y, w, h = ann["bbox"]
            ann["area"] = w * h

        # iscrowd setzen falls fehlt
        ann["iscrowd"] = ann.get("iscrowd", 0)

        fixed_annotations.append(ann)

    data["annotations"] = fixed_annotations

    # --- Speichern ---
    with open(output_json, "w") as f:
        json.dump(data, f)

    print(f"✅ Fixed COCO saved to: {output_json}")


# Run as a script from any working directory.
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]

    fix_coco(
        repo_root / "data/processed/train/annotations.json",
        repo_root / "data/processed/train/annotations_fixed.json",
        prefix="train/images/",
    )

    fix_coco(
        repo_root / "data/processed/val/annotations.json",
        repo_root / "data/processed/val/annotations_fixed.json",
        prefix="val/images/",
    )
