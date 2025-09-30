import kagglehub
import json
import os
import shutil
from pathlib import Path


class InitialConfig:
    def __init__(self, data_path: str = ".", kaggle_json_path: str = None):
        self.dataset = "jessicali9530/stl10"
        self.download_path = os.path.abspath(data_path)
        self.final_path = os.path.join(self.download_path, "stl10")

        os.environ['KAGGLEHUB_CACHE'] = self.download_path

        self._load_credentials(kaggle_json_path)

    def _load_credentials(self, kaggle_json_path: str = None) -> None:
        if kaggle_json_path is None:
            kaggle_json_path = os.path.join(
                Path.home(), '.kaggle', 'kaggle.json'
            )

        if not os.path.exists(kaggle_json_path):
            raise FileNotFoundError(
                f"No se encontró kaggle.json en {kaggle_json_path}.\n"
                "Descárgalo desde https://www.kaggle.com/settings/account "
                "y colócalo en ~/.kaggle/kaggle.json"
            )

        with open(kaggle_json_path, 'r') as f:
            credentials = json.load(f)

        self.username = credentials.get('username')
        self.key = credentials.get('key')

        if not self.username or not self.key:
            raise ValueError(
                "El archivo kaggle.json no contiene 'username' o 'key'")

        os.environ['KAGGLE_USERNAME'] = self.username
        os.environ['KAGGLE_KEY'] = self.key

    def _organize_dataset(self, downloaded_path: str) -> None:
        if os.path.exists(self.final_path):
            shutil.rmtree(self.final_path)

        shutil.copytree(downloaded_path, self.final_path)

        cache_dir = os.path.join(self.download_path, "datasets")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

    def start(self) -> str:
        path = kagglehub.dataset_download(self.dataset)
        print(f"Descargado temporalmente en: {path}")

        self._organize_dataset(path)
        print(f"Dataset organizado en: {self.final_path}")

        return self.final_path
