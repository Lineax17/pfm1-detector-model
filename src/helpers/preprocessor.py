import json
from pathlib import Path

import cv2

from helpers import ImageTransformator


class Preprocessor:
    def __init__(self, config=None):
        self.config = config

    def preprocess(self,
        image_path,
        annotation_path,
        image_size=(320, 320),
        augmentation=False
    ):
        # Load COCO annotations
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

        # Load image and bbox
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = [ann["bbox"] for ann in annotations]
        labels = [ann["category_id"] for ann in annotations]

        # Apply transform

        transformator = ImageTransformator(
            image_size=image_size,
            augmentation=augmentation
        )

        transformed = transformator.transform(
            image=image,
            bboxes=bboxes,
            labels=labels
        )

        return transformed

    def preprocess_folder(
        self,
        folder_path,
        annotation_path,
        image_size=(320, 320),
    ):
        pass
