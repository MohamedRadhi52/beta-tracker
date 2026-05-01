# Évalue le modèle de détection de prises sur le jeu de test Roboflow.
# Usage : python scripts/evaluate.py [--split test|val]
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from beta_tracker_core import HOLD_MODEL_PATH, ROOT_DIR, require_file

DATA_YAML = ROOT_DIR / "Climbing-Hold-Detection-9" / "data.yaml"
RUNS_EVAL_DIR = ROOT_DIR / "runs" / "eval"


def parse_args():
    parser = argparse.ArgumentParser(description="Evalue le modele de detection de prises.")
    parser.add_argument("--split", default="test", choices=["test", "val"], help="Split à évaluer (default: test).")
    parser.add_argument("--save-json", action="store_true", help="Sauvegarde les prédictions au format COCO JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(require_file(HOLD_MODEL_PATH, "Modele de prises")))

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"data.yaml introuvable : {DATA_YAML}")

    print(f"Evaluation sur le split '{args.split}'...")
    metrics = model.val(
        data=str(DATA_YAML),
        split=args.split,
        save_json=args.save_json,
        project=str(RUNS_EVAL_DIR),
        name=f"holds_{args.split}",
        verbose=False,
    )

    # Segmentation et détection ont des attributs différents.
    box = getattr(metrics, "box", None)
    seg = getattr(metrics, "seg", None)

    if box is not None:
        print("\n--- Détection (boîtes) ---")
        print(f"  mAP@50      : {box.map50:.4f}")
        print(f"  mAP@50-95   : {box.map:.4f}")
        print(f"  Précision   : {box.mp:.4f}")
        print(f"  Rappel      : {box.mr:.4f}")

    if seg is not None:
        print("\n--- Segmentation (masques) ---")
        print(f"  mAP@50      : {seg.map50:.4f}")
        print(f"  mAP@50-95   : {seg.map:.4f}")
        print(f"  Précision   : {seg.mp:.4f}")
        print(f"  Rappel      : {seg.mr:.4f}")

    if box is None and seg is None:
        print("Aucune métrique disponible — vérifiez le type de modèle.")

    save_dir = getattr(metrics, "save_dir", RUNS_EVAL_DIR / f"holds_{args.split}")
    print(f"\nRésultats sauvegardés dans {save_dir}")


if __name__ == "__main__":
    main()
