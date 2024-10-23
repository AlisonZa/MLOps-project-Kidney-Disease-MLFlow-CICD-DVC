from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig
import os

class ConfigurationManager:
  
  # Recebe as constantes dos arquivos de configuração e parâmetros
  def __init__(
    self,
    config_filepath = CONFIG_FILE_PATH,
    params_filepath = PARAMS_FILE_PATH,):

    self.config = read_yaml(config_filepath)
    self.params = read_yaml(params_filepath)

    create_directories([self.config.artifacts_root])

  # Recebe as configurações do componente de Ingestão de dados
  def get_data_ingestion_config(self) -> DataIngestionConfig: # dataclass criador por nós em entities
    config = self.config.data_ingestion

    create_directories([config.root_dir])
    data_ingestion_config = DataIngestionConfig(
      root_dir= config.root_dir,
      source_URL= config.source_URL,
      local_data_file= config.local_data_file,
      unzip_dir= config.unzip_dir,
    )

    return data_ingestion_config
  
  # Recebe os parâmetros e configuração do componente de Treinamento do Modelo
  def get_prepare_base_model_config(self) -> PrepareBaseModelConfig: # dataclass criado por nós em entities
    config = self.config.prepare_base_model

    create_directories([config.root_dir])
    prepare_base_model_config = PrepareBaseModelConfig(
      root_dir=Path(config.root_dir), # os caminhos tem que ser passados dentro de PATH
      base_model_path=Path(config.base_model_path),
      updated_base_model_path= Path(config.updated_base_model_path),
      params_image_size= self.params.IMAGE_SIZE, # O que vier do params.yaml é capitalizado, com self.params
      params_learning_rate= self.params.LEARNING_RATE,
      params_include_top = self.params.INCLUDE_TOP,
      params_weights= self.params.WEIGHTS,
      params_classes= self.params.CLASSES,
    )

    return prepare_base_model_config  
  
  # Recebe as configurações do componente de Treinamento
  def get_training_config(self) -> TrainingConfig: # dataclass criado por nós em entities  
    config = self.config.training

    # Isto daqui é diferente posi precisamos acessar algo de outro componenete
    prepare_base_model = self.config.prepare_base_model
    training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")

    create_directories([config.root_dir])
    training_config = TrainingConfig(
      
      # Puxam direto da var config
      root_dir= Path(config.root_dir),
      trained_model_path = Path(config.trained_model_path),

      # Puxam de artefatos
      updated_base_model_path= Path(prepare_base_model.updated_base_model_path),
      training_data= Path(training_data),
      
      # Olhar em params.yaml
      params_epochs= self.params.EPOCHS,
      params_batch_size= self.params.BATCH_SIZE,
      params_is_augumentation= self.params.AUGUMENTATION,
      params_image_size= self.params.IMAGE_SIZE,
    )

    return training_config
  
  def get_evaluation_config(self) -> EvaluationConfig:
      eval_config = EvaluationConfig(
          path_of_model="artifacts/training/trained_model.h5",
          training_data="artifacts/data_ingestion/kidney-ct-scan-image",
          mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", " "),
          all_params=self.params,
          params_image_size=self.params.IMAGE_SIZE,
          params_batch_size=self.params.BATCH_SIZE
      )
      return eval_config
