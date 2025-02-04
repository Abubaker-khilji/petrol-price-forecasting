import pandas as pd
import pickle
from src.componenets.data_ingestion import DataIngestion
from src.componenets.data_transformation import DataPreprocessing
from src.componenets.model_trainer import ModelTraining
from src.logger import log

def train():
    log("Starting Training Pipeline...")

    # Load Data
    ingestion = DataIngestion("dataset/train_data.csv", "dataset/test_data.csv")
    train_data, test_data = ingestion.load_data()
    log("Data Loaded Successfully.")

    # Preprocessing
    train_data.rename(columns={"date": "ds", "petrol": "y"}, inplace=True)
    processor = DataPreprocessing(train_data)
    processed_train = processor.preprocess()
    log("Data Preprocessing Completed.")

    # Train Model
    trainer = ModelTraining(processed_train)
    model = trainer.train_model()
    log("Model Training Completed.")

    # Save Model
    with open("best_prophet_model.pkl", "wb") as f:
        pickle.dump(model, f)
    log("Model Saved Successfully.")

if __name__ == "__main__":
    train()