import sys
import io
import json
import csv
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from beta_tracker_core import (
    ARTICULATIONS,
    COULEURS_PRISES,
    DEFAULT_DISTANCE_MAX_FRAC,
    DEFAULT_HOLD_CONFIDENCE,
    DEFAULT_POSE_CONFIDENCE,
    HOLD_MODEL_PATH,
    POSE_MODEL_PATH,
    detect_contacts,
    require_file,
    signature_contact,
)

st.set_page_config(page_title="Beta-Tracker", layout="wide")

# ─── Couleurs membres ─────────────────────────────────────────────────────────
_COULEURS_MEMBRES = {
    "Poignet Gauche":  (255, 140,   0),
    "Poignet Droit":   ( 30, 144, 255),
    "Cheville Gauche": ( 50, 205,  50),
    "Cheville Droite": (186,  85, 211),
}
_COULEURS_MEMBRES_HEX = {
    k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in _COULEURS_MEMBRES.items()
}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Paramètres")

    seuil_confiance = st.slider(
        "Confiance squelette min.",
        min_value=0.0, max_value=1.0, value=DEFAULT_POSE_CONFIDENCE, step=0.05,
        help="Filtre les points du squelette peu fiables : membre caché, flou, etc."
    )
    distance_max_pct = st.slider(
        "Distance contact max (% diag.)",
        min_value=1, max_value=10, value=int(DEFAULT_DISTANCE_MAX_FRAC * 100), step=1,
        help="Seuil en % de la diagonale de l'image."
    )
    distance_max_frac = distance_max_pct / 100.0

    seuil_confiance_prise = st.slider(
        "Confiance prise min.",
        min_value=0.0, max_value=1.0, value=DEFAULT_HOLD_CONFIDENCE, step=0.05,
        help="Ignore les détections de prises peu sûres."
    )

    st.divider()

    couleur_cible = st.selectbox(
        "Filtre couleur de prise",
        ["toutes"] + COULEURS_PRISES,
        index=0,
    )

    st.divider()

    if st.button("Réinitialiser la séquence", use_container_width=True):
        st.session_state.sequence = []
        st.success("Séquence effacée.")

    st.divider()
    st.caption(
        "**Légende membres**\n\n"
        + "\n".join(
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{hex_};border-radius:50%;margin-right:6px"></span>{nom}'
            for nom, hex_ in _COULEURS_MEMBRES_HEX.items()
        ),
        unsafe_allow_html=True,
    )

# ─── Session state ─────────────────────────────────────────────────────────
if "sequence" not in st.session_state:
    st.session_state.sequence = []

# ─── Modèles ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_prises = YOLO(str(require_file(HOLD_MODEL_PATH, "Modele de prises")))
    model_pose   = YOLO(str(require_file(POSE_MODEL_PATH, "Modele de pose")))
    return model_prises, model_pose

model_prises, model_pose = load_models()

# ─── Rendu annoté ─────────────────────────────────────────────────────────────
def render_annotation(
    image_pil: Image.Image,
    res_prises,
    res_pose,
    contacts: list,
    seuil_conf_prise: float = 0.4,
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
        color = _COULEURS_MEMBRES.get(c["articulation"], (255, 220, 0))

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


# ─── Analyse d'une image ──────────────────────────────────────────────────────
def analyser_image(image: Image.Image, source_label: str) -> tuple[list[dict], np.ndarray, object, object]:
    # Lance les modèles sur une image, retourne les contacts détectés.
    res_prises = model_prises(image)[0]
    res_pose   = model_pose(image)[0]
    contacts   = detect_contacts(
        res_prises, res_pose,
        seuil_confiance=seuil_confiance,
        seuil_confiance_prise=seuil_confiance_prise,
        couleur_cible=couleur_cible,
        image=image,
        distance_max_frac=distance_max_frac,
    )
    img_ann = render_annotation(image, res_prises, res_pose, contacts, seuil_confiance_prise)
    return contacts, img_ann, res_prises, res_pose


def ajouter_a_sequence(contacts: list[dict], source: str):
    etape_base = len(st.session_state.sequence) + 1
    for i, c in enumerate(contacts):
        st.session_state.sequence.append({
            "etape":         etape_base + i,
            "articulation":  c["articulation"],
            "couleur_prise": c["couleur_prise"],
            "confiance":     c["confiance_pose"],
            "source":        source,
        })


def afficher_contacts(contacts: list[dict]):
    for c in contacts:
        hex_color = _COULEURS_MEMBRES_HEX.get(c["articulation"], "#ccc")
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{hex_color};border-radius:50%;margin-right:6px"></span>'
            f"**{c['articulation']}** touche une prise **{c['couleur_prise']}** "
            f"| pose: `{c['confiance_pose']:.2f}` | prise: `{c['confiance_prise']:.2f}` "
            f"| dist: `{c['distance_px']:.1f}px`",
            unsafe_allow_html=True,
        )


# ─── Export ───────────────────────────────────────────────────────────────────
def export_json(sequence: list) -> str:
    return json.dumps(
        {"beta": sequence, "export": datetime.now().isoformat()},
        ensure_ascii=False, indent=2,
    )

def export_csv(sequence: list) -> str:
    if not sequence:
        return ""
    output = io.StringIO()
    fieldnames = list(sequence[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sequence)
    return output.getvalue()


# ─── Header ───────────────────────────────────────────────────────────────────
st.title("Beta-Tracker - Analyse d'escalade")
st.caption("Détection des prises, posture du grimpeur et contacts main/pied sur image ou vidéo.")

tab_images, tab_video, tab_sequence = st.tabs(["Images", "Vidéo", "Séquence et export"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Images
# ══════════════════════════════════════════════════════════════════════════════
with tab_images:
    uploaded_files = st.file_uploader(
        "Ajoutez une ou plusieurs images dans l'ordre chronologique",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="L'ordre d'upload définit l'ordre de la séquence.",
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} image(s) sélectionnée(s). Lancez l'analyse quand tout est prêt.")

        if st.button("Analyser les images", type="primary", key="btn_analyse_images"):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.markdown(f"---\n#### {uploaded_file.name}")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, use_container_width=True, caption="Original")

                with col2:
                    with st.spinner(f"Analyse de {uploaded_file.name}..."):
                        contacts, img_ann, _, _ = analyser_image(image, uploaded_file.name)

                    st.image(img_ann, use_container_width=True, caption="Image annotée")

                    if not contacts:
                        st.warning("Aucun contact détecté. Ajustez les paramètres si besoin.")
                    else:
                        st.success(f"{len(contacts)} contact(s) détecté(s)")
                        afficher_contacts(contacts)
                        ajouter_a_sequence(contacts, uploaded_file.name)

        if st.session_state.sequence:
            st.success(f"Séquence mise à jour : {len(st.session_state.sequence)} mouvement(s) au total.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Vidéo
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown(
        "Ajoutez une vidéo pour analyser les mouvements image par image. "
        "Vous pouvez analyser seulement une image sur N pour aller plus vite."
    )

    video_file = st.file_uploader(
        "Vidéo d'escalade",
        type=["mp4", "mov", "avi"],
        key="video_upload",
    )

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        frame_skip = st.number_input(
            "Analyser une image sur", min_value=1, max_value=60, value=10,
            help="Avec 10, le traitement analyse une image sur 10."
        )
    with col_v2:
        max_frames = st.number_input(
            "Limite d'images analysées (0 = tout)",
            min_value=0, max_value=10000, value=200,
            help="Pour tester rapidement. 0 = vidéo complète."
        )

    if video_file and st.button("Lancer l'analyse vidéo", type="primary", key="btn_analyse_video"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.mp4"
            output_path = Path(tmp_dir) / "input_annotated.mp4"
            input_path.write_bytes(video_file.getbuffer())

            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                st.error("Impossible d'ouvrir la vidéo.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps          = cap.get(cv2.CAP_PROP_FPS) or 25
                width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                st.info(f"Vidéo : {width}x{height} @ {fps:.1f} fps - {total_frames} frames totales")

                out_fps = max(fps / frame_skip, 1.0)
                out = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    out_fps,
                    (width, height),
                )

                if not out.isOpened():
                    cap.release()
                    st.error("Impossible de créer la vidéo annotée.")
                else:
                    progress_bar  = st.progress(0, text="Traitement en cours...")
                    status_txt    = st.empty()
                    contacts_video = []
                    contacts_events = []
                    active_signatures = {}
                    frame_idx     = 0
                    analysed      = 0
                    limite        = max_frames if max_frames > 0 else float("inf")

                    try:
                        while cap.isOpened() and analysed < limite:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_idx += 1
                            if frame_idx % frame_skip != 0:
                                continue
                            analysed += 1

                            img_pil    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            res_prises = model_prises(img_pil, verbose=False)[0]
                            res_pose   = model_pose(img_pil, verbose=False)[0]

                            contacts = detect_contacts(
                                res_prises, res_pose,
                                seuil_confiance=seuil_confiance,
                                seuil_confiance_prise=seuil_confiance_prise,
                                couleur_cible=couleur_cible,
                                image=img_pil,
                                distance_max_frac=distance_max_frac,
                            )

                            img_ann = render_annotation(img_pil, res_prises, res_pose, contacts, seuil_confiance_prise)
                            out.write(cv2.cvtColor(img_ann, cv2.COLOR_RGB2BGR))

                            current_articulations = set()
                            for c in contacts:
                                contact_entry = {
                                    **c,
                                    "frame":   frame_idx,
                                    "temps_s": round(frame_idx / fps, 2),
                                }
                                contacts_video.append(contact_entry)
                                current_articulations.add(c["articulation"])

                                signature = signature_contact(contact_entry)
                                if active_signatures.get(c["articulation"]) != signature:
                                    contacts_events.append(contact_entry)
                                    active_signatures[c["articulation"]] = signature

                            for articulation in ARTICULATIONS:
                                if articulation not in current_articulations:
                                    active_signatures.pop(articulation, None)

                            prog = min(analysed / (limite if max_frames > 0 else (total_frames / frame_skip + 1)), 1.0)
                            progress_bar.progress(prog, text=f"Image {frame_idx} / {total_frames}...")
                            status_txt.text(
                                f"Analysées : {analysed} | Contacts bruts : {len(contacts_video)} "
                                f"| Mouvements : {len(contacts_events)}"
                            )
                    finally:
                        cap.release()
                        out.release()
                        progress_bar.empty()
                        status_txt.empty()

                    st.success(
                        f"{analysed} image(s) analysée(s) - {len(contacts_video)} contacts bruts, "
                        f"{len(contacts_events)} mouvement(s) ajoutable(s)."
                    )

                    if output_path.is_file():
                        st.download_button(
                            "Télécharger la vidéo annotée",
                            data=output_path.read_bytes(),
                            file_name="beta_annotated.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )

                    # Timeline contacts vidéo
                    if contacts_video:
                        st.markdown("#### Contacts détectés dans la vidéo")
                        df_vid = pd.DataFrame([{
                            "Temps (s)":    c["temps_s"],
                            "Image":        c["frame"],
                            "Membre":       c["articulation"],
                            "Couleur":      c["couleur_prise"],
                            "Conf. pose":   c["confiance_pose"],
                            "Conf. prise":  c["confiance_prise"],
                        } for c in contacts_video])
                        st.dataframe(df_vid, use_container_width=True)

                    if contacts_events:
                        etape_base = len(st.session_state.sequence) + 1
                        for i, c in enumerate(contacts_events):
                            st.session_state.sequence.append({
                                "etape":         etape_base + i,
                                "articulation":  c["articulation"],
                                "couleur_prise": c["couleur_prise"],
                                "confiance":     c["confiance_pose"],
                                "source":        f"image {c['frame']} ({c['temps_s']}s)",
                            })

                        st.info(f"{len(contacts_events)} mouvement(s) ajouté(s) à la séquence.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Séquence & Export
# ══════════════════════════════════════════════════════════════════════════════
with tab_sequence:
    sequence = st.session_state.sequence

    if not sequence:
        st.info("Aucune séquence enregistrée. Analysez des images ou une vidéo.")
    else:
        st.subheader(f"Béta reconstruit - {len(sequence)} mouvement(s)")

        # ── Métriques rapides ────────────────────────────────────────────────
        membres_counter = Counter(m["articulation"] for m in sequence)
        plus_actif, nb_plus_actif = membres_counter.most_common(1)[0]
        conf_moy = float(np.mean([m["confiance"] for m in sequence]))

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Mouvements", len(sequence))
        col_m2.metric("Membres distincts", len(membres_counter))
        col_m3.metric("Membre le + actif", plus_actif, f"{nb_plus_actif} fois")
        col_m4.metric("Confiance moy.", f"{conf_moy:.2f}")

        st.divider()

        # ── Timeline visuelle ────────────────────────────────────────────────
        st.markdown("#### Timeline du béta")
        for m in sequence:
            hex_color  = _COULEURS_MEMBRES_HEX.get(m["articulation"], "#999")
            source_str = f' <span style="color:#888;font-size:0.8em">({m.get("source", "")})</span>' if m.get("source") else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                f'  <span style="background:{hex_color};color:white;font-weight:700;'
                f'    padding:3px 10px;border-radius:12px;min-width:32px;text-align:center;'
                f'    font-size:0.85em;">{m["etape"]}</span>'
                f'  <span><b>{m["articulation"]}</b>'
                f'    touche une prise <b style="color:{hex_color}">{m["couleur_prise"]}</b>'
                f'    <span style="color:#888;font-size:0.85em">conf. {m["confiance"]:.2f}</span>'
                f'    {source_str}'
                f'  </span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Graphiques ───────────────────────────────────────────────────────
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("#### Contacts par membre")
            df_membres = pd.DataFrame(
                list(membres_counter.items()), columns=["Membre", "Contacts"]
            ).set_index("Membre")
            st.bar_chart(df_membres)

        with col_g2:
            st.markdown("#### Contacts par couleur de prise")
            couleurs_counter = Counter(m["couleur_prise"] for m in sequence)
            df_couleurs = pd.DataFrame(
                list(couleurs_counter.items()), columns=["Couleur", "Contacts"]
            ).set_index("Couleur")
            st.bar_chart(df_couleurs)

        st.divider()

        # ── Tableau complet ──────────────────────────────────────────────────
        with st.expander("Voir le tableau complet"):
            st.dataframe(
                pd.DataFrame(sequence),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # ── Export ───────────────────────────────────────────────────────────
        st.markdown("#### Exporter le béta")
        col_j, col_c = st.columns(2)
        with col_j:
            st.download_button(
                "Télécharger JSON",
                data=export_json(sequence),
                file_name="beta_sequence.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_c:
            st.download_button(
                "Télécharger CSV",
                data=export_csv(sequence),
                file_name="beta_sequence.csv",
                mime="text/csv",
                use_container_width=True,
            )
