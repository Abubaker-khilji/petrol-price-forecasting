from prophet import Prophet
import pickle

class ModelTraining:
    def _init_(self, data):
        self.data = data

    def train_model(self):
        model = Prophet(seasonality_mode="additive")
        model.fit(self.data)
        return model

if __name__ == "_main_":
    import pandas as pd
    data = pd.read_csv("dataset/train_data.csv")
    data.rename(columns={"date": "ds", "petrol": "y"}, inplace=True)

    trainer = ModelTraining(data)
    trained_model = trainer.train_model()

    with open("best_prophet_model.pkl", "wb") as f:
        pickle.dump(trained_model, f)
    print("Model Trained and Saved Successfully!")