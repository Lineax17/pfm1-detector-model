from pathlib import Path

from helpers.preprocessor import Preprocessor
from helpers.visualizer import Visualizer

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = REPO_ROOT / "data" / "original" / "mine" / "obvious" / "vid-mine-obv-11" / "images" / "default" / "frame_000010.png"
ANNOTATION_PATH = REPO_ROOT / "data" / "original" / "mine" / "obvious" / "vid-mine-obv-11" / "annotations" / "instances_default.json"

def preprocess_test(
    image_path,
    annotation_path,
    image_size=(320, 320),
    augmentation=False
):

    # Show original
    visualizer = Visualizer()
    visualizer.show_bbox(image_path, annotation_path, "Original")

    # Apply transform
    preprocessor = Preprocessor()

    transformed = preprocessor.preprocess(
        image_path,
        annotation_path,
        image_size,
        augmentation
    )

    transformed_image = transformed["image"]
    transformed_boxes = transformed["bboxes"]

    # Show transformed
    visualizer.show_bbox(transformed_image, transformed_boxes, "Transformed")


if __name__ == "__main__":

    preprocess_test(
        image_path=IMAGE_PATH,
        annotation_path=ANNOTATION_PATH,
        image_size=(320, 320),
        augmentation=False
    )