from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger
from dotenv import load_dotenv
import os

STAGE_NAME = "Evaluate the model Logs in MLFlow"

load_dotenv()
os.environ["MLFLOW_TRACKING_URI"]=os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"]= os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"]= os.getenv("MLFLOW_TRACKING_PASSWORD")

class EvaluationPipeline():
    def __init__(self):
        pass

    # The source cames from the component file
    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow() 
        # uncomment to log into mlflow
     
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage: {STAGE_NAME} started <<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>> stage: {STAGE_NAME} Completed <<<<\n\n x================x")
    except Exception as e:
        logger.exception(e)
        raise e