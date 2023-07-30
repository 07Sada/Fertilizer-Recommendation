from src.logger import logging
from src.exception import FertilizerException
from src.utils import get_collection_as_dataframe
from src.entity import config_entity
from src.entity import artifact_entity
import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

def start_training_pipeline():
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        # data ingestion
        data_ingestion_config = config_entity.DataIngestionConfig(
            training_pipeline_config=training_pipeline_config)

        data_ingestion_config.to_dict()

        data_ingestion = DataIngestion(
            data_ingestion_config=data_ingestion_config)

        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    
        print(f"Data Ingestin complete")

        # data validation 
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)

        data_validation = DataValidation(data_validation_config=data_validation_config, 
                                        data_ingestion_arfitact=data_ingestion_artifact)
        
        data_validation.initiate_data_validation()
        print(f"Data Validation Complete")

        # data transformation 
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)

        data_transformation = DataTransformation(data_transformation_config=data_transformation_config, 
                                                data_ingestion_artifact=data_ingestion_artifact)

        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(f"Data Transformation Complete")                                                

        # model trainer
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)

        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, 
                                    data_transformation_artifact=data_transformation_artifact)

        model_trainer_artifact = model_trainer.initial_model_trainer()
        print(f"Model Trainer Complete")

        # model evaluation 
        model_evaluation_config = config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)

        model_evaluation = ModelEvaluation(
            model_eval_config = model_evaluation_config, 
            data_ingestion_artifact = data_ingestion_artifact, 
            data_transformation_artifact = data_transformation_artifact, 
            model_trainer_artifact = model_trainer_artifact)

        model_evalution_artifact = model_evaluation.initiate_model_evaluation()
        print(f"Model Evaluation Complete")

        # model pusher
        model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)

        model_pusher = ModelPusher(
            model_pusher_config = model_pusher_config, 
            data_transformation_artifact = data_transformation_artifact, 
            model_trainer_artifact = model_trainer_artifact)

        model_trainer_artifact = model_pusher.initiate_model_pusher()
        print(f"Model Pusher Complete")

    except Exception as e:
        raise FertilizerException(e, sys)

