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
    """
    Plot a single YOLO format image with bounding boxes.

    Args:
        yolo_root: Root directory of YOLO dataset
        split: Dataset split (train/val/test)
        image_name: Name of the image to plot
        output_dir: Directory to save visualization (if not displaying)
        display: If True, try to display with cv2.imshow(); else save to file
    """
    img_path = yolo_root / split / "images" / image_name
    label_path = yolo_root / split / "labels" / (Path(image_name).stem + ".txt")

    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return

    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Failed to read image: {img_path}")
        return

    h, w = img.shape[:2]
    bbox_count = 0

    # Read and plot bounding boxes
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                cls = int(parts[0])
                x, y, bw, bh = map(float, parts[1:5])

                # Convert from YOLO format (normalized) to pixel coordinates
                x_center = x * w
                y_center = y * h
                box_w = bw * w
                box_h = bh * h

                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)

                # Draw rectangle and class label
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
        print(f"✅ Found {bbox_count} bounding boxes")
    else:
        print(f"⚠️ No labels found for: {image_name}")

    # Display or save image
    if display:
        try:
            cv2.imshow("YOLO Sample", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(f"✅ Image displayed")
        except cv2.error as e:
            print(f"⚠️ Display not available: {e}")
            print(f"Saving to file instead...")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{Path(image_name).stem}_annotated.png"
            cv2.imwrite(str(output_path), img)
            print(f"✅ Image saved to: {output_path}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_name).stem}_annotated.png"
        cv2.imwrite(str(output_path), img)
        print(f"✅ Image saved to: {output_path}")


if __name__ == "__main__":
    plot_yolo_bbox()
