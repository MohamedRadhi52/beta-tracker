import sys
from pathlib import Path

from ultralytics import YOLO
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beta_tracker_core import ROOT_DIR

DATA_YAML = ROOT_DIR / "Climbing-Hold-Detection-9" / "data.yaml"

wandb.init(project="Beta_Tracker_Holds", name="yolov8s-seg-entrainement-long")

model = YOLO("yolov8s-seg.pt")

results = model.train(
    data=str(DATA_YAML),
    epochs=150,
    imgsz=640,
    batch=16,
    project="wandb",
    patience=30
)

wandb.finish()