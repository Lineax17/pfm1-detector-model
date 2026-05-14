from pathlib import Path

from helpers.visualizer import Visualizer

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = REPO_ROOT / "data" / "original" / "mine" / "obvious" / "vid-mine-obv-11" / "images" / "default" / "frame_000010.png"
ANNOTATION_PATH = REPO_ROOT / "data" / "original" / "mine" / "obvious" / "vid-mine-obv-11" / "annotations" / "instances_default.json"

def show_bbox():
    visualizer = Visualizer()
    visualizer.show_bbox(IMAGE_PATH, ANNOTATION_PATH)