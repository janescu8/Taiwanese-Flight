import streamlit as st
import pandas as pd
import joblib

# Load the synthetic dataset
data_path = "taiwanese_flight_data_2010_2019.csv"
data = pd.read_csv(data_path)

# Load your model
model_path = "linear_regression_flight_fare.pkl"
model_data = joblib.load(model_path)
model = model_data["model"]
label_encoders = model_data["label_encoders"]

st.title("Flight Fare Prediction")

# Select flight based on synthetic data
st.sidebar.header("Select Flight Details")

selected_airline = st.sidebar.selectbox("Airline", options=data["Airline"].unique())
selected_source = st.sidebar.selectbox("Source", options=data["Source"].unique())
selected_destination = st.sidebar.selectbox("Destination", options=data["Destination"].unique())
selected_date = st.sidebar.date_input("Date of Journey")

data_filtered = data[
    (data["Airline"] == selected_airline)
    & (data["Source"] == selected_source)
    & (data["Destination"] == selected_destination)
]

if not data_filtered.empty:
    selected_flight = st.selectbox(
        "Select a flight:",
        options=data_filtered["Route"].unique(),
    )

    if st.button("Predict Fare"):
        # Prepare input data for the model
        flight_data = data_filtered[data_filtered["Route"] == selected_flight].iloc[0]

        # Encode features for the model
        model_input = [
            label_encoders["Airline"].transform([selected_airline])[0],
            label_encoders["Source"].transform([selected_source])[0],
            label_encoders["Destination"].transform([selected_destination])[0],
            label_encoders["Route"].transform([selected_flight])[0],
            label_encoders["Total_Stops"].transform([flight_data["Total_Stops"]])[0],
        ]

        # Predict fare
        prediction = model.predict([model_input])[0]
        st.success(f"Predicted Fare: ₹{prediction:.2f}")
else:
    st.warning("No flights found matching the selected criteria.")
