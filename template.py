import os, sys
from pathlib import Path
import logging

while True:
    project_name = input("Enter your project name: ")
    if project_name !="":
        break

# src/__init__.py
# src/compontes/__init__.py
list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/config.py",
    f"{project_name}/app.py",
    f"{project_name}/logger.py",
    f"{project_name}/exception.py",
    f"{project_name}/setup.py",
    f"{project_name}/utils.py",
    f"{project_name}/predictor.py",
    "main.py",
]


for filepth in list_of_files:
    filepath = Path(filepth)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass

    else:
        logging.info("file is already present at : {filepath}")