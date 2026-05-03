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
    COULEURS_MEMBRES,
    COULEURS_PRISES,
    DEFAULT_DISTANCE_MAX_FRAC,
    DEFAULT_HOLD_CONFIDENCE,
    DEFAULT_POSE_CONFIDENCE,
    HOLD_MODEL_PATH,
    POSE_MODEL_PATH,
    detect_contacts,
    render_annotation,
    require_file,
    signature_contact,
)

st.set_page_config(page_title="Beta-Tracker", layout="wide")

# couleurs en hex pour streamlit
_COULEURS_MEMBRES_HEX = {
    k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in COULEURS_MEMBRES.items()
}

# sidebar
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

# init du state
if "sequence" not in st.session_state:
    st.session_state.sequence = []

# chargement des modèles en cache
@st.cache_resource
def load_models():
    model_prises = YOLO(str(require_file(HOLD_MODEL_PATH, "Modele de prises")))
    model_pose = YOLO(str(require_file(POSE_MODEL_PATH, "Modele de pose")))
    return model_prises, model_pose

model_prises, model_pose = load_models()

# helpers
def analyser_image(image: Image.Image, source_label: str) -> tuple[list[dict], np.ndarray, object, object]:
    res_prises = model_prises(image)[0]
    res_pose = model_pose(image)[0]
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
            "etape": etape_base + i,
            "articulation": c["articulation"],
            "couleur_prise": c["couleur_prise"],
            "confiance": c["confiance_pose"],
            "source": source,
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


# exports
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


st.title("Beta-Tracker - Analyse d'escalade")
st.caption("Détection des prises, posture du grimpeur et contacts main/pied sur image ou vidéo.")

tab_images, tab_video, tab_sequence = st.tabs(["Images", "Vidéo", "Séquence et export"])


# tab images
with tab_images:
    uploaded_files = st.file_uploader(
        "Importez vos images (l'ordre compte)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} image(s) sélectionnée(s).")

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
            st.success(f"Séquence maj : {len(st.session_state.sequence)} mouvement(s).")


# tab video
with tab_video:
    st.markdown(
        "Import vidéo. Jouez sur le frame skip pour accélérer l'analyse."
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
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
                    progress_bar = st.progress(0, text="Traitement en cours...")
                    status_txt = st.empty()
                    contacts_video = []
                    events = []
                    active_sigs = {}
                    frame_idx = 0
                    analysed = 0
                    limite = max_frames if max_frames > 0 else float("inf")

                    try:
                        while cap.isOpened() and analysed < limite:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_idx += 1
                            if frame_idx % frame_skip != 0:
                                continue
                            analysed += 1

                            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            res_prises = model_prises(img_pil, verbose=False)[0]
                            res_pose = model_pose(img_pil, verbose=False)[0]

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
                                    "frame": frame_idx,
                                    "temps_s": round(frame_idx / fps, 2),
                                }
                                contacts_video.append(contact_entry)
                                current_articulations.add(c["articulation"])

                                signature = signature_contact(contact_entry)
                                if active_sigs.get(c["articulation"]) != signature:
                                    events.append(contact_entry)
                                    active_sigs[c["articulation"]] = signature

                            for articulation in ARTICULATIONS:
                                if articulation not in current_articulations:
                                    active_sigs.pop(articulation, None)

                            prog = min(analysed / (limite if max_frames > 0 else (total_frames / frame_skip + 1)), 1.0)
                            progress_bar.progress(prog, text=f"Image {frame_idx} / {total_frames}...")
                            status_txt.text(
                                f"Analysées : {analysed} | Contacts bruts : {len(contacts_video)} "
                                f"| Mouvements : {len(events)}"
                            )
                    finally:
                        cap.release()
                        out.release()
                        progress_bar.empty()
                        status_txt.empty()

                    st.success(
                        f"{analysed} image(s) analysée(s) - {len(contacts_video)} contacts bruts, "
                        f"{len(events)} mouvement(s) trouvé(s)."
                    )

                    if output_path.is_file():
                        st.download_button(
                            "Télécharger la vidéo annotée",
                            data=output_path.read_bytes(),
                            file_name="beta_annotated.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )

                # affichage timeline video
                    if contacts_video:
                        st.markdown("#### Contacts détectés dans la vidéo")
                        df_vid = pd.DataFrame([{
                            "Temps (s)": c["temps_s"],
                            "Image": c["frame"],
                            "Membre": c["articulation"],
                            "Couleur": c["couleur_prise"],
                            "Conf. pose": c["confiance_pose"],
                            "Conf. prise": c["confiance_prise"],
                        } for c in contacts_video])
                        st.dataframe(df_vid, use_container_width=True)

                    if events:
                        etape_base = len(st.session_state.sequence) + 1
                        for i, c in enumerate(events):
                            st.session_state.sequence.append({
                                "etape":         etape_base + i,
                                "articulation": c["articulation"],
                                "couleur_prise": c["couleur_prise"],
                                "confiance": c["confiance_pose"],
                                "source": f"image {c['frame']} ({c['temps_s']}s)",
                            })

                        st.info(f"{len(events)} mouvement(s) ajouté(s).")


# tab export sequence
with tab_sequence:
    sequence = st.session_state.sequence

    if not sequence:
        st.info("Aucune séquence enregistrée. Analysez des images ou une vidéo.")
    else:
        st.subheader(f"Béta reconstruit - {len(sequence)} mouvement(s)")

        # stats rapides
        membres_counter = Counter(m["articulation"] for m in sequence)
        plus_actif, nb_plus_actif = membres_counter.most_common(1)[0]
        conf_moy = float(np.mean([m["confiance"] for m in sequence]))

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Mouvements", len(sequence))
        col_m2.metric("Membres distincts", len(membres_counter))
        col_m3.metric("Membre le + actif", plus_actif, f"{nb_plus_actif} fois")
        col_m4.metric("Confiance moy.", f"{conf_moy:.2f}")

        st.divider()

        # timeline
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

        # graphes
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

        # recap
        with st.expander("Voir le tableau complet"):
            st.dataframe(
                pd.DataFrame(sequence),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # actions d'export
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
