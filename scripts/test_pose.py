import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beta_tracker_core import DEFAULT_IMAGE_PATH, POSE_MODEL_PATH, require_file


def parse_args():
    parser = argparse.ArgumentParser(description="Teste le modele de pose sur une image.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH), help="Image a analyser.")
    parser.add_argument("--show", action="store_true", help="Affiche une fenetre OpenCV si disponible.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = require_file(args.image, "Image de test")
    model_pose = YOLO(require_file(POSE_MODEL_PATH, "Modele de pose"))

    print("Lancement de l'analyse de posture...")
    results = model_pose.predict(source=str(image_path), show=args.show, save=True)

    for result in results:
        nb_personnes = 0 if result.keypoints is None else len(result.keypoints)
        print(f"{nb_personnes} personne(s) détectée(s).")


if __name__ == "__main__":
    main()
