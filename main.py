from config.InitialConfig import InitialConfig


if __name__ == "__main__":
    config = InitialConfig(kaggle_json_path="./kaggle.json", data_path=".")
    config.start()
