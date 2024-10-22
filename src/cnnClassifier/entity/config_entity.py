from dataclasses import dataclass
from pathlib import Path

# Definimos nossas entidades com o que foi configurado no primeiro bloco do arquivo yaml
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path