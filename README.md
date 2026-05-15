# Detection of PFM1 Landmine

This repository contains code for the detection of PFM1 landmines using machine learning techniques. The project aims to develop a model that can accurately identify the presence of PFM1 landmines in various environments.

## Docker training (ROCm)

Trainingsartefakte werden nach `models/` geschrieben. Mount das Verzeichnis, um `.pt`-Gewichte aus dem Container zu erhalten.

```bash
docker build -t pfm1-detector .

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  -v "$(pwd)/data:/workspace/pfm1-detector-model/data" \
  -v "$(pwd)/models:/workspace/pfm1-detector-model/models" \
  pfm1-detector
```

Die exportierten Gewichte liegen danach in `models/pfm1-yolo11s-best.pt` (und optional `models/pfm1-yolo11s-last.pt`).
