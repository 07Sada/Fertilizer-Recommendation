from src.entity import config_entity 
from src.entity import artifact_entity
from src.logger import logging 
from src.exception import FertilizerException
from src import utils

from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 
import sys 
import os

class DataIngestion:

    def __init__(self, data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            logging.info(f"\n\n{'>'*50} Data Ingestion {'<'*50}\n")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise FertilizerException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data as pandas Dataframe ")

            df: pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name, 
                collection_name=self.data_ingestion_config.collection_name)

            logging.info(f"Saving data in feature store")

            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            logging.info(f"Saving dataframe into feature store")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,
                        index=False,
                        header=True)
                
            logging.info(f"Split the dataset into train and test")
            train_df, test_df = train_test_split(
                df, test_size=self.data_ingestion_config.test_size, random_state=42
            )

            logging.info(f"Create dataset directory if not available")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info(f"Save df to feature store folder")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,
                            index=False,
                            header=True)
            
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,
                            index=False,
                            header=True)

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path, 
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data Ingestion Completed. Artifacts saved")
            
            return data_ingestion_artifact
        
        except Exception as e:
            raise FertilizerException(e, sys)