import json
import shutil
from pathlib import Path

import cv2

from src.helpers.preprocessor import Preprocessor

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_folder_processing_test():
    source_images = (
        REPO_ROOT
        / "data"
        / "original"
        / "mine"
        / "obvious"
        / "vid-mine-obv-11"
        / "images"
        / "default"
    )
    annotation_path = (
        REPO_ROOT
        / "data"
        / "original"
        / "mine"
        / "obvious"
        / "vid-mine-obv-11"
        / "annotations"
        / "instances_default.json"
    )

    input_dir = REPO_ROOT / "src" / "test" / "folder_processing_input"
    output_dir = REPO_ROOT / "src" / "test" / "folder_processing_test"

    if input_dir.exists():
        shutil.rmtree(input_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    input_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(source_images.iterdir()):
        if image_path.is_file():
            shutil.copy(image_path, input_dir / image_path.name)

    Preprocessor().preprocess_folder(
        folder_path=input_dir,
        target_path=output_dir,
        annotation_path=annotation_path,
        image_size=(320, 320),
    )

    output_image = output_dir / "images" / "frame_000010.png"
    output_annotation = output_dir / "annotations" / annotation_path.name

    assert output_image.exists(), "Processed image was not written."
    assert output_annotation.exists(), "Processed annotations.json was not written."

    processed = cv2.imread(str(output_image))
    assert processed is not None, "Processed image could not be read."
    assert processed.shape[0] == 320 and processed.shape[1] == 320, "Image size is not 320x320."

    with open(output_annotation, "r") as f:
        coco = json.load(f)

    assert coco.get("images"), "No images found in output annotations."
    assert coco.get("annotations") is not None, "Annotations field missing in output."


if __name__ == "__main__":
    run_folder_processing_test()
    print("folder_processing_test: OK")
