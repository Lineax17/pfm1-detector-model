from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="/mnt/data/Work/Repos/pfm1-detector-model/src/training/data.yml",
    epochs=200,
    imgsz=1024,
    batch=32,
    device=0,
)