from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"

HOLD_MODEL_PATH = MODELS_DIR / "holds_model.pt"
POSE_MODEL_PATH = MODELS_DIR / "yolov8n-pose.pt"
DEFAULT_IMAGE_PATH = ASSETS_DIR / "grimpeur.png"
DEFAULT_VIDEO_PATH = ASSETS_DIR / "climb.mp4"
DEFAULT_VIDEO_OUTPUT_PATH = ASSETS_DIR / "resultat_beta_tracker.mp4"

DEFAULT_POSE_CONFIDENCE = 0.5
DEFAULT_HOLD_CONFIDENCE = 0.4
DEFAULT_DISTANCE_MAX_FRAC = 0.05
DEFAULT_SURFACE_MIN_HOLD = 250

ARTICULATIONS = {
    "Poignet Gauche": 9,
    "Poignet Droit": 10,
    "Cheville Gauche": 15,
    "Cheville Droite": 16,
}

COULEURS_PRISES = ["rouge", "orange", "jaune", "verte", "bleue", "violette", "rose", "noire", "blanche", "grise"]


def require_file(path: str | Path, description: str) -> Path:
    resolved_path = Path(path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = ROOT_DIR / resolved_path
    resolved_path = resolved_path.resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"{description} introuvable: {resolved_path}")
    return resolved_path


def est_en_contact_ellipse(px: float, py: float, boite: Iterable[float], marge: float = 0.2) -> bool:
    # Detecte si un point est dans une ellipse centree sur la boite de la prise.
    # La marge reduit la zone utile pour limiter les faux contacts en bordure.
    x_min, y_min, x_max, y_max = [float(value) for value in boite]
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    rayon_x = (x_max - x_min) / 2 * (1 - marge)
    rayon_y = (y_max - y_min) / 2 * (1 - marge)

    if rayon_x <= 0 or rayon_y <= 0:
        return False

    return ((px - cx) ** 2 / rayon_x**2 + (py - cy) ** 2 / rayon_y**2) <= 1.0


def get_type_prise(resultats_prises, idx: int) -> str:
    try:
        cls_id = int(resultats_prises.boxes.cls[idx].item())
        return resultats_prises.names.get(cls_id, "inconnue")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return "inconnue"


def _distance_point_boite(px: float, py: float, boite: Iterable[float]) -> float:
    x_min, y_min, x_max, y_max = [float(value) for value in boite]
    dx = max(x_min - px, 0, px - x_max)
    dy = max(y_min - py, 0, py - y_max)
    return float((dx**2 + dy**2) ** 0.5)


def _distance_point_centre(px: float, py: float, boite: Iterable[float]) -> float:
    x_min, y_min, x_max, y_max = [float(value) for value in boite]
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    return float(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5)


def signature_contact(contact: dict, spatial_bucket: float = 24.0) -> tuple:
    # Signature stable pour dedupliquer les contacts video en evenements de beta.
    # Le bucket spatial evite qu'une boite quasi identique cree un nouvel evenement.
    x_min, y_min, x_max, y_max = [float(value) for value in contact["boite"]]
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    return (
        contact["articulation"],
        contact.get("couleur_prise", "inconnue"),
        contact.get("type_prise", "inconnue"),
        round(cx / spatial_bucket),
        round(cy / spatial_bucket),
    )


def _rgb_to_hsv(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = pixels.astype(np.float32) / 255.0
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    maximum = pixels.max(axis=1)
    minimum = pixels.min(axis=1)
    delta = maximum - minimum

    hue = np.zeros_like(maximum)
    masque = delta > 1e-6
    r_max = masque & (maximum == r)
    g_max = masque & (maximum == g)
    b_max = masque & (maximum == b)
    hue[r_max] = (60 * ((g[r_max] - b[r_max]) / delta[r_max]) + 360) % 360
    hue[g_max] = 60 * ((b[g_max] - r[g_max]) / delta[g_max]) + 120
    hue[b_max] = 60 * ((r[b_max] - g[b_max]) / delta[b_max]) + 240

    saturation = np.zeros_like(maximum)
    saturation[maximum > 1e-6] = delta[maximum > 1e-6] / maximum[maximum > 1e-6]
    return hue, saturation, maximum


def _nom_couleur_depuis_hue(hue: float) -> str:
    if hue < 15 or hue >= 345:
        return "rouge"
    if hue < 48:
        return "orange"
    if hue < 78:
        return "jaune"
    if hue < 165:
        return "verte"
    if hue < 255:
        return "bleue"
    if hue < 300:
        return "violette"
    return "rose"


def estimer_couleur_prise(image, resultats_prises, idx: int) -> tuple[str, float]:
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image_array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    hauteur, largeur = image_array.shape[:2]

    masque = None
    if resultats_prises.masks is not None:
        masque = resultats_prises.masks.data[idx].cpu().numpy().astype(bool)
        if masque.shape != (hauteur, largeur):
            masque = np.asarray(Image.fromarray(masque.astype(np.uint8) * 255).resize((largeur, hauteur))) > 0

    if masque is not None and masque.any():
        pixels = image_array[masque]
    else:
        x_min, y_min, x_max, y_max = [int(round(value)) for value in resultats_prises.boxes.xyxy[idx].tolist()]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(largeur, x_max), min(hauteur, y_max)
        pixels = image_array[y_min:y_max, x_min:x_max].reshape(-1, 3)

    if len(pixels) == 0:
        return "inconnue", 0.0

    hue, saturation, value = _rgb_to_hsv(pixels)
    nb_pixels = len(pixels)

    masque_noir = value < 0.22
    masque_blanc = (value > 0.72) & (saturation < 0.18)
    masque_gris = (value >= 0.22) & (saturation < 0.16)
    neutres = {
        "noire": float(masque_noir.mean()),
        "blanche": float(masque_blanc.mean()),
        "grise": float(masque_gris.mean()),
    }
    couleur_neutre, ratio_neutre = max(neutres.items(), key=lambda item: item[1])
    if ratio_neutre > 0.6:
        return couleur_neutre, round(ratio_neutre, 2)

    masque_colore = (saturation > 0.22) & (value > 0.18)
    if masque_colore.sum() < max(8, nb_pixels * 0.08):
        return couleur_neutre, round(ratio_neutre, 2)

    votes = {couleur: 0 for couleur in COULEURS_PRISES if couleur not in {"noire", "blanche", "grise"}}
    for h in hue[masque_colore]:
        votes[_nom_couleur_depuis_hue(float(h))] += 1

    couleur, nb_votes = max(votes.items(), key=lambda item: item[1])
    confiance = nb_votes / int(masque_colore.sum())
    return couleur, round(float(confiance), 2)


def detect_contacts(
    resultats_prises,
    resultats_pose,
    seuil_confiance: float = DEFAULT_POSE_CONFIDENCE,
    distance_max_px: float = 18.0,
    seuil_confiance_prise: float = DEFAULT_HOLD_CONFIDENCE,
    surface_min_prise: float = DEFAULT_SURFACE_MIN_HOLD,
    couleur_cible: str | None = None,
    articulations: dict[str, int] | None = None,
    image=None,
    distance_max_frac: float | None = None,
) -> list[dict]:
    # distance_max_frac: si fourni, remplace distance_max_px par cette fraction de la diagonale
    # de l'image (ex: 0.05 = 5%). S'adapte automatiquement à toute résolution.
    articulations = articulations or ARTICULATIONS
    contacts = []

    # Normalise le seuil de distance par rapport à la taille de l'image.
    # distance_max_frac permet de passer un seuil indépendant de la résolution.
    distance_max_eff = distance_max_px
    if distance_max_frac is not None:
        orig_shape = getattr(resultats_pose, "orig_shape", None)
        if orig_shape:
            h, w = orig_shape
            distance_max_eff = distance_max_frac * (h**2 + w**2) ** 0.5

    if resultats_pose.keypoints is None or len(resultats_pose.keypoints.data) == 0:
        return contacts

    donnees_pose = resultats_pose.keypoints.data[0]
    boites = resultats_prises.boxes.xyxy

    for nom, idx_kp in articulations.items():
        point = donnees_pose[idx_kp]
        px, py, confiance = float(point[0]), float(point[1]), float(point[2])

        if confiance < seuil_confiance or (px == 0 and py == 0):
            continue

        meilleur_contact = None
        for i_boite, boite in enumerate(boites):
            confiance_prise = float(resultats_prises.boxes.conf[i_boite])
            if confiance_prise < seuil_confiance_prise:
                continue

            x_min, y_min, x_max, y_max = [float(value) for value in boite.tolist()]
            surface = (x_max - x_min) * (y_max - y_min)
            if surface < surface_min_prise:
                continue

            distance_boite = _distance_point_boite(px, py, boite.tolist())
            if distance_boite > distance_max_eff:
                continue

            couleur, confiance_couleur = (
                estimer_couleur_prise(image, resultats_prises, i_boite) if image is not None else ("inconnue", 0.0)
            )
            if couleur_cible and couleur_cible != "toutes" and couleur != couleur_cible:
                continue

            distance_centre = _distance_point_centre(px, py, boite.tolist())
            # Le bonus de confiance est borné à 30% de distance_max_eff pour rester proportionnel.
            score = distance_boite + distance_centre * 0.05 - confiance_prise * (distance_max_eff * 0.3)
            if meilleur_contact is None or score < meilleur_contact["score"]:
                meilleur_contact = {
                    "score": score,
                    "articulation": nom,
                    "couleur_prise": couleur,
                    "confiance_couleur": confiance_couleur,
                    "type_prise": get_type_prise(resultats_prises, i_boite),
                    "confiance": round(confiance, 2),
                    "confiance_pose": round(confiance, 2),
                    "confiance_prise": round(confiance_prise, 2),
                    "distance_px": round(distance_boite, 1),
                    "point": (round(px, 1), round(py, 1)),
                    "boite": [round(float(value), 1) for value in boite.tolist()],
                }

        if meilleur_contact is not None:
            meilleur_contact.pop("score")
            contacts.append(meilleur_contact)

    return contacts
