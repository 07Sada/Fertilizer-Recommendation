from src.entity import config_entity
from src.entity import artifact_entity
from src.logger import logging 
from src.exception import FertilizerException
from src.config import TARGET_COLUMN
from src import utils

from typing import Optional
from scipy.stats import ks_2samp
import pandas as pd 
import numpy as np 
import os
import sys

class DataValidation:

    def __init__(
        self, 
        data_validation_config: config_entity.DataValidationConfig,
        data_ingestion_arfitact: artifact_entity.DataIngestionArtifact
        ):
    
        try:
            logging.info(f"\n\n{'>'*50} Data Validation Initiated {'<'*50}\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_arfitact = data_ingestion_arfitact
            self.validation_error = dict()
        
        except Exception as e:
            raise FertilizerException(e, sys) 

    def is_required_colums_exists(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        report_key_name: str
    ) -> bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: {base_column} is not available")
                    missing_columns.append(base_column)

            if len(missing_columns) > 0:
                self.validation_error[report_key_name] = missing_columns
                return False

            return True
    
        except Exception as e:
            raise FertilizerException(e, sys)

    def data_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        report_key_name: str
    ):
        try:
            drift_report = dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]

                # Null hypothesis is that both column data drawn from same distribution

                logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype}")
                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue > 0.05:
                    # we are accepting the null hypothesis
                    drift_report[base_column] = {
                        "pvalue": float(same_distribution.pvalue),
                        "same_distribution":True
                    }
                
                else:
                    drift_report[base_column] = {
                        "pvalue": float(same_distribution.pvalue),
                        "same_distribution":False
                    }
            self.validation_error[report_key_name] = drift_report

        except Exception as e:
            raise FertilizerException(e, sys)
    
    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)

            logging.info(f"Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_arfitact.train_file_path)

            logging.info(f"Reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_arfitact.test_file_path)

            exclude_column = [TARGET_COLUMN]
            base_df = utils.seperate_dependant_column(df=base_df, exclude_column=exclude_column)
            train_df = utils.seperate_dependant_column(df=train_df, exclude_column=exclude_column)
            test_df = utils.seperate_dependant_column(df=test_df, exclude_column=exclude_column)

            logging.info(f"Is all required columns present in the train_df")
            train_df_columns_status = self.is_required_colums_exists(
                base_df=base_df, 
                current_df=train_df, 
                report_key_name='missing_columns_within_train_dataset')
            
            test_df_columns_status = self.is_required_colums_exists(
                base_df=base_df, 
                current_df=test_df, 
                report_key_name='missing_columns_within_test_dataset')

            if train_df_columns_status:
                logging.info(f"As all colum are availabel in train_df hence detecting data drift")

                self.data_drift(base_df=base_df, current_df=train_df, report_key_name='data_drift_within_train_dataset')

            if test_df_columns_status:
                logging.info(f"As all columns are availabel in test_df hence detecting data drift")

                self.data_drift(base_df=base_df, current_df=test_df, report_key_name='data_drift_within_test_dataset')

            # writting the report 
            logging.info(f"Writing report in yaml format")
            utils.write_yaml_file(
                file_path=self.data_validation_config.report_file_path, 
                data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path)

            logging.info(f"Data Vadidation Completed. Artifacts saved")

            return data_validation_artifact
        
        except Exception as e:
            raise FertilizerException(e, sys)