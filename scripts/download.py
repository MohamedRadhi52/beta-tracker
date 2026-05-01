import os

from roboflow import Roboflow


api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("Definissez la variable d'environnement ROBOFLOW_API_KEY avant de telecharger le dataset.")

rf = Roboflow(api_key=api_key)

# Ciblage du dataset Climbtag
# On utilise l'URL : universe.roboflow.com/climbtag/climbing-hold-detection-7uehq
project = rf.workspace("climbtag").project("climbing-hold-detection-7uehq")

# Choix de la version et téléchargement
# Ce projet a 9 versions, on cible la version la plus aboutie.
version = project.version(9) 

# On télécharge au format YOLOv8 (cela va formater les polygones correctement pour YOLOv8-seg)
dataset = version.download("yolov8")

print(f"Dataset téléchargé ici : {dataset.location}")
