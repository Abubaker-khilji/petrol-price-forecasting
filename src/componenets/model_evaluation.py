import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelEvaluation:
    def _init_(self, model, test_data):
        self.model = model
        self.test_data = test_data

    def evaluate(self):
        future = self.model.make_future_dataframe(periods=len(self.test_data))
        forecast = self.model.predict(future)
        y_true = self.test_data['petrol']
        y_pred = forecast['yhat'][:len(y_true)]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)

        return mae, rmse

if __name__ == "__main__":
    test_data = pd.read_csv("dataset/test_data.csv")
    test_data.rename(columns={"date": "ds", "petrol": "y"}, inplace=True)

    with open("best_prophet_model.pkl", "rb") as f:
        model = pickle.load(f)

    evaluator = ModelEvaluation(model, test_data)
    mae, rmse = evaluator.evaluate()
    print(f"MAE: {mae}, RMSE: {rmse}")