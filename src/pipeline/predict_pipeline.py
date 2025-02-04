import pandas as pd
import pickle
from src.logger import log

class PredictionPipeline:
    def _init_(self, model_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, periods=30):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast

if __name__ == "_main_":
    log("Starting Prediction Pipeline...")
    
    # Load Model
    model_path = "best_prophet_model.pkl"
    predictor = PredictionPipeline(model_path)

    # Predict Future Prices
    forecast = predictor.predict(periods=30)
    forecast.to_csv("petrol_price_forecast.csv", index=False)
    
    log("Predictions Saved to 'petrol_price_forecast.csv'.")