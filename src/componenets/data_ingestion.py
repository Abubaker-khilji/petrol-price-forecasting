import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException  # Corrected spelling
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Read the dataset
            file = pd.read_csv(r'notebook/data/dataset/train_data.csv')  # Fixed path

            # Save the raw dataset
            file.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved successfully")

            # Split the dataset
            train_set, test_set = train_test_split(file, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train and test data saved successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:  # Use Exception instead of CustomException
            logging.error(f"Error in Data Ingestion: {str(e)}")
            raise CustomException(str(e), sys)  # Properly raising CustomException
