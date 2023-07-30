import pandas as pd
from src.logger import logging
from src.exception import FertilizerException
from src.config import mongo_client
import os
import sys
import numpy as np
import yaml
import dill

def get_collection_as_dataframe(
    database_name: str, collection_name: str
) -> pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(
            f"Reading data from database: {database_name} and collection: {collection_name}"
        )
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"{database_name} found in the mongodb")

        if "_id" in df.columns:
            logging.info("Dropping column: '_id'")
            df = df.drop(columns=["_id"], axis=1)
        logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        raise FertilizerException(e, sys)


def seperate_dependant_column(df: pd.DataFrame, exclude_column: list) -> pd.DataFrame:
    final_dataframe = df.drop(exclude_column, axis=1)

    return final_dataframe


def get_column_indices(numerical_features: list, categorical_features: list, base_file_path: str):

    dataset = pd.read_csv(base_file_path)

    numerical_feature_indices = [dataset.columns.get_loc(feature) for feature in numerical_features]
    categorical_feature_indices = [dataset.columns.get_loc(feature) for feature in categorical_features]

    return numerical_feature_indices, categorical_feature_indices


def write_yaml_file(file_path, data: dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)

        with open(file_path, "w") as file_writer:
            yaml.dump(data, file_writer)
    except Exception as e:
        raise FertilizerException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save object method of utils")
    except Exception as e:
        raise FertilizerException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise FertilizerException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    save numpy array data to file
    file_path : str location of the file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_ojb:
            np.save(file_obj, array)

    except Exception as e:
        raise FertilizerException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)

    except Exception as e:
        raise CropException(e, sys)
