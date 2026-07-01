"""Convert a PyTorch .pt model to TensorFlow Lite (.tflite) format."""

from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = REPO_ROOT / "models" / "pfm1-yolo11s-v2.pt"


def convert_pt_to_tflite(model_path: str | Path) -> Path:
    """Convert an Ultralytics YOLO .pt model to TFLite format.

    The resulting .tflite file is saved in the same directory as the .pt model.

    Parameters
    ----------
    model_path : str | Path
        Path to the .pt model file.

    Returns
    -------
    Path
        Path to the generated .tflite file.
    """
    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if model_path.suffix != ".pt":
        raise ValueError("Input file must be a .pt model.")

    model = YOLO(str(model_path))

    # Export creates a subdirectory named after the model; move .tflite next to the source .pt
    exported_path = Path(model.export(format="tflite"))

    target_path = model_path.with_suffix(".tflite")
    exported_path.replace(target_path)

    return target_path


if __name__ == "__main__":
    convert_pt_to_tflite(MODEL)
