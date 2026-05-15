from pathlib import Path
import shutil
import tempfile

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "src" / "training" / "data.yml"
DATA_ROOT = REPO_ROOT / "data" / "processed"
RUNS_DIR = REPO_ROOT / "runs" / "detect"
MODELS_DIR = REPO_ROOT / "models"
LOCAL_WEIGHTS = REPO_ROOT / "yolo11s.pt"


def _write_data_yaml(data_yaml: Path, data_root: Path) -> Path:
    content = data_yaml.read_text(encoding="utf-8").splitlines()
    updated = []
    replaced = False
    for line in content:
        if line.strip().startswith("path:"):
            updated.append(f"path: {data_root}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.insert(0, f"path: {data_root}")
    tmp_dir = Path(tempfile.mkdtemp(prefix="pfm1-data-"))
    tmp_path = tmp_dir / "data.yml"
    tmp_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return tmp_path


def _export_weights(run_dir: Path, models_dir: Path) -> None:
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    models_dir.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        shutil.copy2(best_pt, models_dir / "pfm1-yolo11s-best.pt")
    if last_pt.exists():
        shutil.copy2(last_pt, models_dir / "pfm1-yolo11s-last.pt")


model_path = str(LOCAL_WEIGHTS) if LOCAL_WEIGHTS.exists() else "yolo11s.pt"
model = YOLO(model_path)

patched_yaml = _write_data_yaml(DATA_YAML, DATA_ROOT)

model.train(
    data=str(patched_yaml),
    epochs=200,
    imgsz=1024,
    batch=32,
    device=0,
    project=str(RUNS_DIR),
    name="train",
    exist_ok=True,
)

_export_weights(RUNS_DIR / "train", MODELS_DIR)
