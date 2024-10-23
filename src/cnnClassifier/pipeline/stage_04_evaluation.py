from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger

STAGE_NAME = "Evaluate the model Logs in MLFlow"


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
     
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage: {STAGE_NAME} started <<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>> stage: {STAGE_NAME} Completed <<<<\n\n x================x")
    except Exception as e:
        logger.exception(e)
        raise e