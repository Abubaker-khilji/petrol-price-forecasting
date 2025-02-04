import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def _init_(self, df):
        self.df = df

    def preprocess(self):
        self.df.dropna(inplace=True)
        scaler = StandardScaler()
        self.df[['petrol']] = scaler.fit_transform(self.df[['petrol']])
        return self.df

if __name__ == "__main__":
    data = pd.read_csv("dataset/train_data.csv")
    processor = DataPreprocessing(data)
    processed_data = processor.preprocess()
    print("Data Preprocessed Successfully!")