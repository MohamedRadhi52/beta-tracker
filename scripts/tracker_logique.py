import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beta_tracker_core import (
    DEFAULT_DISTANCE_MAX_FRAC,
    DEFAULT_IMAGE_PATH,
    DEFAULT_HOLD_CONFIDENCE,
    DEFAULT_POSE_CONFIDENCE,
    HOLD_MODEL_PATH,
    POSE_MODEL_PATH,
    detect_contacts,
    require_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test de la logique Beta-Tracker sur une image.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH), help="Image a analyser.")
    parser.add_argument("--seuil-pose", default=DEFAULT_POSE_CONFIDENCE, type=float, help="Confiance pose min.")
    parser.add_argument("--seuil-prise", default=DEFAULT_HOLD_CONFIDENCE, type=float, help="Confiance prise min.")
    parser.add_argument("--dist-frac", default=DEFAULT_DISTANCE_MAX_FRAC, type=float, help="Distance contact max en fraction de diagonale.")
    return parser.parse_args()


def main():
    args = parse_args()

    model_prises = YOLO(str(require_file(HOLD_MODEL_PATH, "Modele de prises")))
    model_pose = YOLO(str(require_file(POSE_MODEL_PATH, "Modele de pose")))

    image_path = require_file(args.image, "Image de test")

    print("Détection des prises...")
    res_prises = model_prises(str(image_path), verbose=False)[0]
    print(f"   Prises detectees : {len(res_prises.boxes)}")

    print("Détection du grimpeur...")
    res_pose = model_pose(str(image_path), verbose=False)[0]
    nb_personnes = 0 if res_pose.keypoints is None else len(res_pose.keypoints)
    print(f"   Personnes detectees : {nb_personnes}")

    contacts = detect_contacts(
        res_prises,
        res_pose,
        seuil_confiance=args.seuil_pose,
        seuil_confiance_prise=args.seuil_prise,
        image=image_path,
        distance_max_frac=args.dist_frac,
    )

    if not contacts:
        print("Aucun contact detecte. Essayez par exemple --dist-frac 0.08 si l'image est large ou eloignee.")
    else:
        print(f"Contacts detectes : {len(contacts)}")
        for contact in contacts:
            print(
                "   CONTACT : "
                f"{contact['articulation']} touche une prise {contact['couleur_prise']} "
                f"(pose {contact['confiance_pose']:.2f}, prise {contact['confiance_prise']:.2f}, "
                f"distance {contact['distance_px']:.1f}px)"
            )


if __name__ == "__main__":
    main()
