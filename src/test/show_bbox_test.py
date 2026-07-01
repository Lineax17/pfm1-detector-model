"""Display a single processed image with its COCO bounding boxes."""

from pathlib import Path

from helpers.visualizer import Visualizer

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "train"
    / "images"
    / "vid_mine_obv_33_frame_000050.png"
)
ANNOTATION_PATH = (
    REPO_ROOT / "data" / "processed" / "train" / "annotations.json"
)


def show_bbox():
    visualizer = Visualizer()
    visualizer.show_bbox(IMAGE_PATH, ANNOTATION_PATH)


if __name__ == "__main__":
    show_bbox()
