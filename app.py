import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly
from prophet import Prophet
import datetime

# Load Model
@st.cache_resource
def load_model():
    with open("notebook\\data\\best_prophet_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Streamlit UI
st.title("Petrol Price Forecasting")
st.write("Forecast future petrol prices using a trained Prophet model.")

# User Input: Forecast Period
period = st.slider("Select Forecast Period (Days)", min_value=7, max_value=365, value=30, step=7)

# Prediction
if st.button("Generate Forecast"):
    st.subheader(f"Petrol Price Forecast for Next {period} Days")
    
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    
    # Display Forecast Table
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period))

    # Plot Forecast
    st.subheader("Forecast Plot")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig)

    # Save Forecast
    forecast.to_csv("petrol_price_forecast.csv", index=False)
    st.success("Predictions saved as 'petrol_price_forecast.csv'.")

# File Upload for Custom Predictions
st.subheader("Upload Custom Data for Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    
    if "ds" in user_data.columns:
        predictions = model.predict(user_data)
        st.write(predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # Plot Custom Predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(predictions["ds"], predictions["yhat"], label="Predicted Price", color="blue")
        ax.fill_between(predictions["ds"], predictions["yhat_lower"], predictions["yhat_upper"], alpha=0.2)
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
    else:
        st.error("Uploaded file must contain a 'ds' column with dates.")

st.write("Developed by Abubakar")