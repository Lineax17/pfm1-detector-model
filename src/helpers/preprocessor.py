"""Preprocess images and COCO annotations: resize, pad, and augment."""

import json
from pathlib import Path

import cv2

from helpers import ImageTransformator


class Preprocessor:
    """Apply image transformations to single images or entire dataset folders."""

    def __init__(self, config=None):
        self.config = config

    def preprocess(
        self,
        image_path,
        annotation_path,
        image_size=(320, 320),
        augmentation=False,
    ):
        """Load a single image, look up its COCO annotations, and transform.

        Parameters
        ----------
        image_path : str | Path
            Path to the image file.
        annotation_path : str | Path
            Path to the COCO JSON annotation file.
        image_size : tuple[int, int]
            Target (height, width).
        augmentation : bool
            Enable random augmentations.

        Returns
        -------
        dict
            Transformed image, bboxes, and labels.
        """
        with open(annotation_path) as f:
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

        # Load image and extract annotations
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = [ann["bbox"] for ann in annotations]
        labels = [ann["category_id"] for ann in annotations]

        transformator = ImageTransformator(
            image_size=image_size, augmentation=augmentation
        )
        transformed = transformator.transform(
            image=image, bboxes=bboxes, labels=labels
        )
        return transformed

    def preprocess_folder(
        self,
        folder_path,
        target_path,
        annotation_path,
        image_size=(320, 320),
    ):
        """Process every image in a folder, saving resized copies and a new COCO JSON.

        Parameters
        ----------
        folder_path : str | Path
            Directory containing source images.
        target_path : str | Path
            Output directory (images/ sub-folder created automatically).
        annotation_path : str | Path
            Original COCO JSON covering all images in the folder.
        image_size : tuple[int, int]
            Target (height, width).
        """
        folder_path = Path(folder_path)
        target_path = Path(target_path)
        annotation_path = Path(annotation_path)

        target_images_dir = target_path / "images"
        target_annotations_dir = target_path
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_annotations_dir.mkdir(parents=True, exist_ok=True)

        with open(annotation_path) as f:
            coco = json.load(f)

        # Index images by filename and annotations by image ID
        image_by_name = {
            img["file_name"]: img for img in coco.get("images", [])
        }
        annotations_by_image_id: dict = {}
        for ann in coco.get("annotations", []):
            annotations_by_image_id.setdefault(ann["image_id"], []).append(ann)

        new_coco = {
            "licenses": coco.get("licenses", []),
            "info": coco.get("info", {}),
            "categories": coco.get("categories", []),
            "images": [],
            "annotations": [],
        }

        transformator = ImageTransformator(
            image_size=image_size, augmentation=False
        )

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        annotation_id = 1

        for item in sorted(folder_path.iterdir()):
            if not item.is_file() or item.suffix.lower() not in image_extensions:
                continue

            image_info = image_by_name.get(item.name)
            if not image_info:
                continue

            image = cv2.imread(str(item))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_id = image_info["id"]
            annotations = annotations_by_image_id.get(image_id, [])
            bboxes = [ann["bbox"] for ann in annotations]
            labels = [ann["category_id"] for ann in annotations]

            transformed = transformator.transform(
                image=image, bboxes=bboxes, labels=labels
            )

            transformed_image = transformed["image"]
            transformed_bboxes = transformed["bboxes"]
            transformed_labels = transformed["labels"]

            # Save transformed image
            target_image_path = target_images_dir / item.name
            bgr_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(target_image_path), bgr_image)

            height, width = transformed_image.shape[:2]
            new_coco["images"].append({
                "id": image_id,
                "file_name": item.name,
                "width": int(width),
                "height": int(height),
            })

            for bbox, label in zip(transformed_bboxes, transformed_labels):
                x, y, w, h = bbox
                new_coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                })
                annotation_id += 1

        output_annotation_path = target_annotations_dir / annotation_path.name
        with open(output_annotation_path, "w") as f:
            json.dump(new_coco, f)
