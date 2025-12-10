from modules.utils.config import AppConfig as Config
from app.ui.gradio_app import launch

if __name__ == "__main__":
    Config.load()
    launch()
