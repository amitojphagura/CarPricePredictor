import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    raw_source_path: str = os.path.join("data", "carcsvdata", "car details v4.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replicates the cleaning / feature-engineering steps worked out in
        notebooks/exploration.ipynb:
          - drop 'Model' (redundant with 'Make') and 'Engine' (redundant with
            the power/torque columns)
          - drop rows with any missing values
          - collapse Length/Width/Height into a single 'Volume' feature
          - parse 'Max Power' / 'Max Torque' free-text columns into numeric
            Power_bhp, Power_rpm, Torque_Nm, Torque_rpm columns
          - drop the rows/columns that couldn't be parsed
        """
        try:
            df = df.drop(columns=["Model", "Engine"])
            df = df.dropna()

            df["Volume"] = (df["Length"] * df["Width"] * df["Height"]) / 1000
            df = df.drop(columns=["Length", "Width", "Height"])

            df["Power_bhp"] = df["Max Power"].str.extract(r"(\d+\.?\d*)\s*bhp").astype(float)
            df["Power_rpm"] = df["Max Power"].str.extract(r"@\s*(\d+\.?\d*)\s*rpm").astype(float)
            df["Torque_Nm"] = df["Max Torque"].str.extract(r"(\d+\.?\d*)\s*Nm").astype(float)
            df["Torque_rpm"] = df["Max Torque"].str.extract(r"@\s*(\d+\.?\d*)\s*rpm").astype(float)

            df = df.dropna(subset=["Power_bhp", "Power_rpm", "Torque_Nm", "Torque_rpm"])
            df = df.drop(columns=["Max Power", "Max Torque"])

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv(self.ingestion_config.raw_source_path)
            logging.info(f"Read the raw dataset as a dataframe, shape={df.shape}")

            df = self.clean_data(df)
            logging.info(f"Cleaned dataset, shape={df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
