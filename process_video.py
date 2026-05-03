# script d'analyse vidéo CLI
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

from beta_tracker_core import (
    ARTICULATIONS,
    DEFAULT_DISTANCE_MAX_FRAC,
    DEFAULT_HOLD_CONFIDENCE,
    DEFAULT_POSE_CONFIDENCE,
    DEFAULT_VIDEO_OUTPUT_PATH,
    DEFAULT_VIDEO_PATH,
    HOLD_MODEL_PATH,
    POSE_MODEL_PATH,
    detect_contacts,
    render_annotation,
    require_file,
    signature_contact,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse une vidéo d'escalade avec Beta-Tracker.")
    parser.add_argument("--input", default=str(DEFAULT_VIDEO_PATH), help="Chemin de la vidéo source.")
    parser.add_argument("--output", default=str(DEFAULT_VIDEO_OUTPUT_PATH), help="Chemin de la vidéo annotée.")
    parser.add_argument("--skip", default=1, type=int, help="Analyser 1 image video sur N (défaut : 1).")
    parser.add_argument("--seuil-pose", default=DEFAULT_POSE_CONFIDENCE, type=float, help="Confiance pose min.")
    parser.add_argument("--seuil-prise", default=DEFAULT_HOLD_CONFIDENCE, type=float, help="Confiance prise min.")
    parser.add_argument("--dist-frac", default=DEFAULT_DISTANCE_MAX_FRAC, type=float, help="Distance contact max en fraction de diagonale.")
    parser.add_argument("--max-frames", default=0, type=int, help="Limite d'images analysees (0 = toute la video).")
    parser.add_argument("--sequence-json", default=None, help="Chemin optionnel pour exporter contacts bruts et mouvements dedupliques.")
    args = parser.parse_args()

    if args.skip < 1:
        parser.error("--skip doit etre superieur ou egal a 1.")
    if args.max_frames < 0:
        parser.error("--max-frames doit etre positif ou nul.")
    if not 0 <= args.seuil_pose <= 1:
        parser.error("--seuil-pose doit etre entre 0 et 1.")
    if not 0 <= args.seuil_prise <= 1:
        parser.error("--seuil-prise doit etre entre 0 et 1.")
    if args.dist_frac <= 0:
        parser.error("--dist-frac doit etre strictement positif.")

    return args


def main():
    args = parse_args()
    video_path  = require_file(args.input, "Vidéo source")
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_prises = YOLO(str(require_file(HOLD_MODEL_PATH, "Modele de prises")))
    model_pose = YOLO(str(require_file(POSE_MODEL_PATH, "Modele de pose")))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    if width <= 0 or height <= 0:
        raise RuntimeError("Dimensions vidéo invalides.")

    out_fps = max(fps / args.skip, 1.0)
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible de créer la vidéo de sortie : {output_path}")

    print(f"Analyse en cours (skip={args.skip})...")
    frame_idx = 0
    nb_ecrits = 0
    nb_contacts = 0
    raw_contacts = [] if args.sequence_json else None
    events = []
    active_sigs = {}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % args.skip != 0:
                continue
            if args.max_frames and nb_ecrits >= args.max_frames:
                break

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res_prises = model_prises(img_pil, verbose=False)[0]
            res_pose = model_pose(img_pil, verbose=False)[0]

            contacts = detect_contacts(
                res_prises, res_pose,
                seuil_confiance=args.seuil_pose,
                seuil_confiance_prise=args.seuil_prise,
                image=img_pil,
                distance_max_frac=args.dist_frac,
            )
            nb_contacts += len(contacts)
            current_articulations = set()
            for contact in contacts:
                contact_entry = {
                    **contact,
                    "frame": frame_idx,
                    "temps_s": round(frame_idx / fps, 2),
                }
                if raw_contacts is not None:
                    raw_contacts.append(contact_entry)
                current_articulations.add(contact["articulation"])

                signature = signature_contact(contact_entry)
                if active_sigs.get(contact["articulation"]) != signature:
                    events.append(contact_entry)
                    active_sigs[contact["articulation"]] = signature

            for articulation in ARTICULATIONS:
                if articulation not in current_articulations:
                    active_sigs.pop(articulation, None)

            img_ann = render_annotation(img_pil, res_prises, res_pose, contacts, args.seuil_prise)
            out.write(cv2.cvtColor(img_ann, cv2.COLOR_RGB2BGR))
            nb_ecrits += 1

            if nb_ecrits % 50 == 0:
                print(f"  -> {nb_ecrits} frames traitées, {nb_contacts} contacts...")

    finally:
        cap.release()
        out.release()

    print(f"\nTerminé : {output_path}")
    print(f"  Frames : {nb_ecrits} | Contacts bruts : {nb_contacts} | Events : {len(events)}")

    if args.sequence_json:
        sequence_path = Path(args.sequence_json).expanduser()
        if not sequence_path.is_absolute():
            sequence_path = Path.cwd() / sequence_path
        sequence_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input": str(video_path),
            "output": str(output_path),
            "settings": {
                "skip": args.skip,
                "seuil_pose": args.seuil_pose,
                "seuil_prise": args.seuil_prise,
                "dist_frac": args.dist_frac,
                "max_frames": args.max_frames,
            },
            "contacts_bruts": raw_contacts or [],
            "mouvements": events,
        }
        sequence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Sequence JSON : {sequence_path}")


if __name__ == "__main__":
    main()
