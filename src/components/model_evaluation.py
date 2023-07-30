from src.predictor import ModelResolver
from src.entity import config_entity
from src.entity import artifact_entity
from src.logger import logging
from src.exception import FertilizerException
from src.utils import load_object

from src.config import TARGET_COLUMN

from sklearn.metrics import f1_score
import pandas as pd 
import numpy as np 
import os 
import sys 

class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: config_entity.ModelEvaluationConfig,
        data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
        data_transformation_artifact: artifact_entity.DataTransformationArtifact,
        model_trainer_artifact: artifact_entity.ModelTrainerArtifact
    ):

        try:
            logging.info(f"\n\n{'>'*50} Model Evaluation Initiated {'<'*50}\n")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise FertilizerException(e, sys)

    
    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info(f"If the saved model directory contains a model, we will compare which model is best trained:\
                the model from the saved model folder or the new model."
            )

            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_accuracy=None)

                logging.info(f"Model Evaluation Artifacts: {model_eval_artifact}")
                return model_eval_artifact
                
            # finding location of transformer, model, and target encoder
            logging.info(f"Finding location of transformer, model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()

            model_path = self.model_resolver.get_latest_model_path()

            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            # finding the location of previous transfomer, model and target encoder
            logging.info(f"Previous trained objects of transformer, model and target encoder")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            # finding the location of currently trained objects
            logging.info(f"Currently trained model objects")
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)

            current_model = load_object(file_path=self.model_trainer_artifact.model_path)

            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            # fetching the testing data 
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]

            y_true = target_encoder.transform(target_df)

            # accuracy using previous trained model
            input_feature_name = list(transformer.feature_names_in_)
            input_arr = transformer.transform(test_df[input_feature_name])

            y_pred = current_model.predict(input_arr)
            y_true = current_target_encoder.transform(target_df)

            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

            # accuracy using current model 
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr = current_transformer.transform(test_df[input_feature_name])

            y_pred = current_model.predict(input_arr)
            y_true = current_target_encoder.transform(target_df)

            current_model_score = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

            if current_model_score <= previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
                                                                        improved_accuracy = current_model_score - previous_model_score)

            logging.info(f"Model Eval Artifacts generated")
            return model_eval_artifact

        except Exception as e:
            raise FertilizerException(e, sys)                                                                        