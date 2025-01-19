# Petrol Price Forecasting

This project is a data-driven application designed to forecast petrol prices using historical data. It implements a machine learning pipeline for time series forecasting and provides a web-based interface for user interaction.

## Features

- **Data Preprocessing**: Handles raw data to clean and prepare it for analysis.
- **Machine Learning**: Uses the `Prophet` library for time series forecasting.
- **Hyperparameter Tuning**: Fine-tunes model parameters for better accuracy.
- **Custom Exception Handling**: Implements a robust error-handling mechanism.
- **Web Interface**: Built with `Streamlit` for a user-friendly frontend.
- **Modular Design**: Code is structured into reusable modules for scalability.

---

## Project Structure

The directory structure of this project is as follows:

PETROL-PRICE-FORECASTING/
├── notebook data/
│   ├── .ipynb_checkpoints/
│   │   └── petrol forecast-checkpoint.ipynb
│   ├── petrol forecast.ipynb
│   ├── petrol_price_forecast.csv
│
├── dataset/
│   ├── sample_submission.csv
│   ├── test_data.csv
│   ├── train_data.csv
│   ├── best_prophet_model.pkl
│
├── src/
│   ├── __pycache__/
│   ├── components/
│   ├── pipeline/
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── venv/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
