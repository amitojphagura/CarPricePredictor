import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    """Orchestrates the full offline training flow: ingest -> transform -> train."""

    def run(self) -> float:
        try:
            logging.info("Starting training pipeline")

            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

            trainer = ModelTrainer()
            r2_square = trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training pipeline complete. Final test R2: {r2_square:.4f}")
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    score = TrainPipeline().run()
    print(f"Final test R2 score: {score:.4f}")
