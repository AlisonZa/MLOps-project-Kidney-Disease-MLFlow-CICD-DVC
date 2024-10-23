from cnnClassifier.components.model_training import Training
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger

STAGE_NAME = "Train the Model Stage"

class TrainingPipeline():
    def __init__(self):
        pass

    # The source cames from the component file
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_training_config() # pega a configuração
        train_model = Training(config=model_training_config) # Puxa a configuração da variável acima, para realizar os métodos abaixo (baixar e extrair)
        train_model.get_base_model()
        train_model.train_valid_generator()
        train_model.train()
     
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage: {STAGE_NAME} started <<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>> stage: {STAGE_NAME} Completed <<<<\n\n x================x")
    except Exception as e:
        logger.exception(e)
        raise e