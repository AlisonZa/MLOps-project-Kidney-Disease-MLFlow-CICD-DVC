from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from src.cnnClassifier.pipeline.stage_03_model_training import TrainingPipeline
from src.cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline
import os

from dotenv import load_dotenv
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"]=os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"]= os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"]= os.getenv("MLFLOW_TRACKING_PASSWORD")

STAGE_NAME = "Data Ingestion Stage"
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage: {STAGE_NAME} started <<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage: { STAGE_NAME} Completed <<<<\n\n x================x ")
    except Exception as e:
        logger.exception(e)
        raise e
    
STAGE_NAME = "Prepare BaseModel Stage"
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage: {STAGE_NAME} started <<<<")
        prepare_base_model = PrepareBaseModelPipeline()
        prepare_base_model.main()
        logger.info(f">>>> stage: { STAGE_NAME} Completed <<<<\n\n x================x ")
    except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Train the Model Stage"
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage: {STAGE_NAME} started <<<<")
        prepare_base_model = TrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>> stage: { STAGE_NAME} Completed <<<<\n\n x================x ")
    except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Evaluate the model Logs in MLFlow"
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage: {STAGE_NAME} started <<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>> stage: {STAGE_NAME} Completed <<<<\n\n x================x")
    except Exception as e:
        logger.exception(e)
        raise e
