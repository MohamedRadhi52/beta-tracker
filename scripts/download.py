import os

from roboflow import Roboflow


api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("Definissez la variable d'environnement ROBOFLOW_API_KEY avant de telecharger le dataset.")

rf = Roboflow(api_key=api_key)
project = rf.workspace("climbtag").project("climbing-hold-detection-7uehq")
version = project.version(9) 
dataset = version.download("yolov8")

print(f"Dataset téléchargé ici : {dataset.location}")
