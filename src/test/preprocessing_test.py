import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from image_transformator import ImageTransformator

REPO_ROOT = Path(__file__).resolve().parents[2]

def draw_bboxes(image, bboxes):

    img = image.copy()

    for (x, y, w, h) in bboxes:

        cv2.rectangle(
            img,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (0, 255, 0),
            2
        )

    return img


def preprocess_test(
    image_path,
    annotation_path,
    image_size=(320, 320),
    augmentation=False
):

    # =========================
    # Load COCO annotations
    # =========================

    with open(annotation_path, "r") as f:
        coco = json.load(f)

    image_filename = Path(image_path).name

    image_info = next(
        img for img in coco["images"]
        if img["file_name"] == image_filename
    )

    image_id = image_info["id"]

    annotations = [
        ann for ann in coco["annotations"]
        if ann["image_id"] == image_id
    ]

    # =========================
    # Load image and bbox
    # =========================

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = [ann["bbox"] for ann in annotations]
    labels = [ann["category_id"] for ann in annotations]

    # =========================
    # Show original
    # =========================

    original_vis = draw_bboxes(image, bboxes)

    plt.figure(figsize=(10, 10))
    plt.title("Original")
    plt.imshow(original_vis)
    plt.axis("off")
    plt.show()

    # =========================
    # Apply transform
    # =========================

    transformator = ImageTransformator(
        image_size=image_size,
        augmentation=augmentation
    )

    transformed = transformator(
        image=image,
        bboxes=bboxes,
        labels=labels
    )

    transformed_image = transformed["image"]
    transformed_boxes = transformed["bboxes"]

    # =========================
    # Show transformed
    # =========================

    transformed_vis = draw_bboxes(
        transformed_image,
        transformed_boxes
    )

    plt.figure(figsize=(10, 10))
    plt.title("Transformed")
    plt.imshow(transformed_vis)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":

    preprocess_test(
        image_path=REPO_ROOT / "data/vid-mine-obv-1/images/default/frame_000000.png",
        annotation_path=REPO_ROOT / "data/vid-mine-obv-1/annotations/instances-default.json",
        image_size=(320, 320),
        augmentation=False
    )