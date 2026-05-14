import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self):
        pass

    def show_bbox(self, image_path, annotation_path, title="Image"):
        if isinstance(image_path, (str, Path)):
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
        else:
            image = image_path
            bboxes = annotation_path

            if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
                image = image.astype(np.uint8, copy=False)

        image_vis = self.draw_bboxes(image, bboxes)

        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.imshow(image_vis)
        plt.axis("off")
        plt.show()

    @staticmethod
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
