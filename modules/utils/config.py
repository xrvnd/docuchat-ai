# modules/utils/config.py
import os
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

class AppConfig(BaseModel):
    google_project_id: str | None = None
    google_processor_id: str | None = None
    google_location: str | None = None
    google_credentials_path: str | None = None

    chroma_db_dir: str = "data/chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "openai"   # for now using temporary

class Config:
    config: AppConfig | None = None

    @staticmethod
    def load():
        load_dotenv()  # load .env first
        with open("configs/default.yaml", "r") as f:
            yaml_config = yaml.safe_load(f)

        # environment override
        merged = {
            **yaml_config,
            **{k.lower(): v for k, v in os.environ.items() if k.lower() in yaml_config}
        }

        Config.config = AppConfig(**merged)
        return Config.config

    @staticmethod
    def get():
        if Config.config is None:
            return Config.load()
        return Config.config
