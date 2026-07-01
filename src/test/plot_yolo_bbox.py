"""Visualise YOLO-format bounding boxes on a single image."""

import cv2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_ROOT = REPO_ROOT / "data" / "yolo"
SPLIT = "train"
IMAGE_NAME = "vid_mine_obv_33_frame_000010.png"
OUTPUT_DIR = REPO_ROOT / "src" / "test" / "yolo_output"


def plot_yolo_bbox(
    yolo_root: Path = YOLO_ROOT,
    split: str = SPLIT,
    image_name: str = IMAGE_NAME,
    output_dir: Path = OUTPUT_DIR,
    display: bool = False,
):
    """Load a YOLO-format image and its label file, then draw bounding boxes.

    Parameters
    ----------
    yolo_root : Path
        Root of the YOLO dataset.
    split : str
        Dataset split (train / val / test).
    image_name : str
        Filename of the image to visualise.
    output_dir : Path
        Directory where the annotated image is saved.
    display : bool
        If True, show the image in a window instead of saving to file.
    """
    img_path = yolo_root / split / "images" / image_name
    label_path = yolo_root / split / "labels" / (
        Path(image_name).stem + ".txt"
    )

    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    h, w = img.shape[:2]
    bbox_count = 0

    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                cls = int(parts[0])
                x, y, bw, bh = map(float, parts[1:5])

                # Convert YOLO normalised coords to pixel coords
                x_center = x * w
                y_center = y * h
                box_w = bw * w
                box_h = bh * h

                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"cls_{cls}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                bbox_count += 1
        print(f"Found {bbox_count} bounding boxes")
    else:
        print(f"No labels found for: {image_name}")

    if display:
        try:
            cv2.imshow("YOLO Sample", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"Display not available: {e}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir / f"{Path(image_name).stem}_annotated.png"
            )
            cv2.imwrite(str(output_path), img)
            print(f"Saved to: {output_path}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_name).stem}_annotated.png"
        cv2.imwrite(str(output_path), img)
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    plot_yolo_bbox()
