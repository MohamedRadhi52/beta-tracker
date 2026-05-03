from pathlib import Path
from typing import Iterable

import cv2
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

COULEURS_MEMBRES = {
    "Poignet Gauche": (255, 140, 0),
    "Poignet Droit": (30, 144, 255),
    "Cheville Gauche": (50, 205, 50),
    "Cheville Droite": (186, 85, 211),
}


def require_file(path: str | Path, description: str) -> Path:
    resolved_path = Path(path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = ROOT_DIR / resolved_path
    resolved_path = resolved_path.resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"{description} introuvable: {resolved_path}")
    return resolved_path


def est_en_contact_ellipse(px: float, py: float, boite: Iterable[float], marge: float = 0.2) -> bool:
    # verifie si on est dans l'ellipse de la bounding box (marge pour limiter les faux contacts)
    x_min, y_min, x_max, y_max = [float(value) for value in boite]
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    rayon_x = (x_max - x_min) / 2 * (1 - marge)
    rayon_y = (y_max - y_min) / 2 * (1 - marge)

    if rayon_x <= 0 or rayon_y <= 0:
        return False

    return ((px - cx) ** 2 / rayon_x**2 + (py - cy) ** 2 / rayon_y**2) <= 1.0


def get_type_prise(res_prises, idx: int) -> str:
    try:
        cls_id = int(res_prises.boxes.cls[idx].item())
        return res_prises.names.get(cls_id, "inconnue")
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
    # hash/signature du contact pour dedupliquer la timeline (avec un arrondi spatial)
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
    
    cmax = pixels.max(axis=1)
    cmin = pixels.min(axis=1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    masque = delta > 1e-6
    r_max = masque & (cmax == r)
    g_max = masque & (cmax == g)
    b_max = masque & (cmax == b)
    hue[r_max] = (60 * ((g[r_max] - b[r_max]) / delta[r_max]) + 360) % 360
    hue[g_max] = 60 * ((b[g_max] - r[g_max]) / delta[g_max]) + 120
    hue[b_max] = 60 * ((r[b_max] - g[b_max]) / delta[b_max]) + 240

    saturation = np.zeros_like(cmax)
    saturation[cmax > 1e-6] = delta[cmax > 1e-6] / cmax[cmax > 1e-6]
    return hue, saturation, cmax


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


def estimer_couleur_prise(image, res_prises, idx: int) -> tuple[str, float]:
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image_array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    hauteur, largeur = image_array.shape[:2]

    masque = None
    if res_prises.masks is not None:
        masque = res_prises.masks.data[idx].cpu().numpy().astype(bool)
        if masque.shape != (hauteur, largeur):
            masque = np.asarray(Image.fromarray(masque.astype(np.uint8) * 255).resize((largeur, hauteur))) > 0

    if masque is not None and masque.any():
        pixels = image_array[masque]
    else:
        x_min, y_min, x_max, y_max = [int(round(value)) for value in res_prises.boxes.xyxy[idx].tolist()]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(largeur, x_max), min(hauteur, y_max)
        pixels = image_array[y_min:y_max, x_min:x_max].reshape(-1, 3)

    if len(pixels) == 0:
        return "inconnue", 0.0

    hue, saturation, value = _rgb_to_hsv(pixels)
    nb_pixels = len(pixels)

    masque_noir = value < 0.2
    masque_blanc = (value > 0.8) & (saturation < 0.15)
    masque_gris = (value >= 0.2) & (saturation < 0.15)
    neutres = {
        "noire": float(masque_noir.mean()),
        "blanche": float(masque_blanc.mean()),
        "grise": float(masque_gris.mean()),
    }
    couleur_neutre, ratio_neutre = max(neutres.items(), key=lambda item: item[1])
    if ratio_neutre > 0.6:
<<<<<<< HEAD
=======
        return couleur_neutre, round(ratio_neutre, 2)

    masque_colore = (saturation > 0.22) & (value > 0.18)
    if masque_colore.sum() < max(8, nb_pixels * 0.08):
>>>>>>> ac3c555 (beta_tracker)
        return couleur_neutre, round(ratio_neutre, 2)

    masque_colore = (saturation > 0.15) & (value > 0.2)
    
    votes = {couleur: 0 for couleur in COULEURS_PRISES if couleur not in {"noire", "blanche", "grise"}}
    for h in hue[masque_colore]:
        votes[_nom_couleur_depuis_hue(float(h))] += 1

    couleur, nb_votes = max(votes.items(), key=lambda item: item[1])
    confiance = nb_votes / int(masque_colore.sum())
    return couleur, round(float(confiance), 2)


def detect_contacts(
    res_prises,
    res_pose,
    seuil_confiance: float = DEFAULT_POSE_CONFIDENCE,
    distance_max_px: float = 18.0,
    seuil_confiance_prise: float = DEFAULT_HOLD_CONFIDENCE,
    surface_min_prise: float = DEFAULT_SURFACE_MIN_HOLD,
    couleur_cible: str | None = None,
    articulations: dict[str, int] | None = None,
    image=None,
    distance_max_frac: float | None = None,
) -> list[dict]:
    articulations = articulations or ARTICULATIONS
    contacts = []

    # calcul de la distance max selon la def de l'image si demandée
    dist_max = distance_max_px
    if distance_max_frac is not None:
        orig_shape = getattr(res_pose, "orig_shape", None)
        if orig_shape:
            h, w = orig_shape
            dist_max = distance_max_frac * (h**2 + w**2) ** 0.5

    if res_pose.keypoints is None or len(res_pose.keypoints.data) == 0:
        return contacts

    kpts = res_pose.keypoints.data[0]
    boxes = res_prises.boxes.xyxy

    for nom, idx_kp in articulations.items():
        point = kpts[idx_kp]
        px, py, conf = float(point[0]), float(point[1]), float(point[2])

        if conf < seuil_confiance or (px == 0 and py == 0):
            continue

        meilleur_contact = None
        for i_boite, boite in enumerate(boxes):
            conf_prise = float(res_prises.boxes.conf[i_boite])
            if conf_prise < seuil_confiance_prise:
                continue

            x_min, y_min, x_max, y_max = [float(value) for value in boite.tolist()]
            surface = (x_max - x_min) * (y_max - y_min)
            if surface < surface_min_prise:
                continue

            dist_box = _distance_point_boite(px, py, boite.tolist())
            if dist_box > dist_max:
                continue

            couleur, confiance_couleur = (
                estimer_couleur_prise(image, res_prises, i_boite) if image is not None else ("inconnue", 0.0)
            )
            if couleur_cible and couleur_cible != "toutes" and couleur != couleur_cible:
                continue

            dist_center = _distance_point_centre(px, py, boite.tolist())
            # bonus de confiance plafonné à 30%
            score = dist_box + dist_center * 0.05 - conf_prise * (dist_max * 0.3)
            if meilleur_contact is None or score < meilleur_contact["score"]:
                meilleur_contact = {
                    "score": score,
                    "articulation": nom,
                    "couleur_prise": couleur,
                    "confiance_couleur": confiance_couleur,
                    "type_prise": get_type_prise(res_prises, i_boite),
                    "confiance": round(conf, 2),
                    "confiance_pose": round(conf, 2),
                    "confiance_prise": round(conf_prise, 2),
                    "distance_px": round(dist_box, 1),
                    "point": (round(px, 1), round(py, 1)),
                    "boite": [round(float(value), 1) for value in boite.tolist()],
                }

        if meilleur_contact is not None:
            meilleur_contact.pop("score")
            contacts.append(meilleur_contact)

    return contacts


def render_annotation(
    image_pil: Image.Image,
    res_prises,
    res_pose,
    contacts: list,
    seuil_conf_prise: float = DEFAULT_HOLD_CONFIDENCE,
) -> np.ndarray:
    img = np.array(image_pil.convert("RGB"))

    for i, boite in enumerate(res_prises.boxes.xyxy):
        if float(res_prises.boxes.conf[i]) < seuil_conf_prise:
            continue
        x1, y1, x2, y2 = [int(v) for v in boite.tolist()]
        cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

    img = res_pose.plot(img=img, labels=False, line_width=2)

    for c in contacts:
        x1, y1, x2, y2 = [int(v) for v in c["boite"]]
        color = COULEURS_MEMBRES.get(c["articulation"], (255, 220, 0))

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        label = f"{c['articulation']}  {c['couleur_prise']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_y0 = max(th + 10, y1)
        cv2.rectangle(img, (x1, label_y0 - th - 10), (x1 + tw + 6, label_y0), color, -1)
        cv2.putText(
            img, label, (x1 + 3, label_y0 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )

        px, py = int(c["point"][0]), int(c["point"][1])
        cv2.circle(img, (px, py), 8, (255, 255, 255), -1)
        cv2.circle(img, (px, py), 5, color, -1)

    return img
