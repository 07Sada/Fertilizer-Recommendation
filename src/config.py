import pymongo
import pandas as pd
import json
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EnvironmentVariable:
    mongo_db_url = os.getenv("MONGO_URL")


env = EnvironmentVariable()

mongo_client = pymongo.MongoClient(env.mongo_db_url)

TARGET_COLUMN = "Fertilizer Name"
NUMERICAL_FEATURES = ['Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
CATEGORICAL_FEATURES = ['Soil Type', 'Crop Type']
BASE_FILE_PATH = os.path.join("fertilizer-prediction/Fertilizer Prediction.csv")