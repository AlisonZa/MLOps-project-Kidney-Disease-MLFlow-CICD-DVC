from dataclasses import dataclass
from pathlib import Path

# Definimos nossas entidades com o que foi configurado no primeiro bloco do arquivo yaml
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path 
    base_model_path: Path # Input Dir
    updated_base_model_path: Path # Output dir
    params_image_size: list # Training Params
    params_learning_rate: float # Training Params
    params_include_top: bool # Training Params
    params_weights: str # Training Params
    params_classes: int # Training Params


# Os training params estão no params.yaml, e os diretórios estão no config.yaml
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path # Vêm do config.yaml
    trained_model_path: Path # Vêm do config.yaml
    updated_base_model_path: Path
    training_data: Path
    # Daqui para baixo são os parâmetros (que vêm do params.yaml) que são utilizados na hora de usar o .fit
    params_epochs: int
    params_batch_size: int
    params_is_augumentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int