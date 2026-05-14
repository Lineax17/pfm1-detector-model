from pathlib import Path
import json
import shutil


class CocoMerger:

    REPO_ROOT = Path(__file__).resolve().parents[2]
    RAW_DATA_DIR = REPO_ROOT / "data" / "original"
    MERGED_DATA_DIR = REPO_ROOT / "data" / "merged"

    def __init__(
        self,
        raw_data_dir= RAW_DATA_DIR,
        processed_dir= MERGED_DATA_DIR
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)

    def merge(self, folders, output_folder):

        output_dir = self.processed_dir / output_folder
        images_output_dir = output_dir / "images"

        images_output_dir.mkdir(parents=True, exist_ok=True)

        merged_coco = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        image_id = 1
        annotation_id = 1

        categories_set = False

        for folder_name in folders:

            folder_path = self._find_folder(folder_name)

            annotation_path = (
                folder_path
                / "annotations"
                / "instances_default.json"
            )

            images_dir = (
                folder_path
                / "images"
                / "default"
            )

            with open(annotation_path, "r") as f:
                coco = json.load(f)

            if not categories_set:
                merged_coco["categories"] = coco["categories"]
                categories_set = True

            image_id_mapping = {}

            for image in coco["images"]:

                old_image_id = image["id"]

                old_filename = Path(image["file_name"]).name

                safe_folder_name = folder_name.replace("-", "_")

                new_filename = (
                    f"{safe_folder_name}_{old_filename}"
                )

                src_image_path = images_dir / old_filename
                dst_image_path = images_output_dir / new_filename

                shutil.copy2(src_image_path, dst_image_path)

                new_image = image.copy()

                new_image["id"] = image_id
                new_image["file_name"] = new_filename

                merged_coco["images"].append(new_image)

                image_id_mapping[old_image_id] = image_id

                image_id += 1

            for annotation in coco["annotations"]:

                new_annotation = annotation.copy()

                new_annotation["id"] = annotation_id

                new_annotation["image_id"] = (
                    image_id_mapping[
                        annotation["image_id"]
                    ]
                )

                merged_coco["annotations"].append(
                    new_annotation
                )

                annotation_id += 1

        output_annotation_path = (
            output_dir / "annotations.json"
        )

        with open(output_annotation_path, "w") as f:
            json.dump(merged_coco, f, indent=2)

        print(
            f"Merged dataset written to: {output_dir}"
        )

    def _find_folder(self, folder_name):

        matches = list(
            self.raw_data_dir.rglob(folder_name)
        )

        if len(matches) == 0:
            raise FileNotFoundError(
                f"Folder not found: {folder_name}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple folders found for: {folder_name}"
            )

        return matches[0]