import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beta_tracker_core import DEFAULT_IMAGE_PATH, HOLD_MODEL_PATH, require_file


def parse_args():
    parser = argparse.ArgumentParser(description="Teste le modele de detection des prises.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH), help="Image a analyser.")
    parser.add_argument("--show", action="store_true", help="Affiche une fenetre OpenCV si disponible.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = require_file(args.image, "Image de test")
    model = YOLO(require_file(HOLD_MODEL_PATH, "Modele de prises"))

    print("Lancement de l'analyse...")
    results = model.predict(source=str(image_path), show=args.show, save=True)
    for result in results:
        print(f"{len(result.boxes)} prise(s) detectee(s).")


if __name__ == "__main__":
    main()
