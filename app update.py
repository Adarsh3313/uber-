import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚖 Uber Ride Booking Status Prediction")

st.write("Enter ride details to predict booking status.")

# Example input fields — replace with your actual dataset's features
pickup_location = st.text_input("Pickup Location")
drop_location = st.text_input("Drop Location")
ride_distance = st.number_input("Ride Distance (km)", min_value=0.0)
ride_fare = st.number_input("Fare Amount", min_value=0.0)
payment_method = st.selectbox("Payment Method", ["Cash", "Card", "Wallet"])

if st.button("Predict Booking Status"):
    input_df = pd.DataFrame({
        "Pickup Location": [pickup_location],
        "Drop Location": [drop_location],
        "Ride Distance": [ride_distance],
        "Fare Amount": [ride_fare],
        "Payment Method": [payment_method]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Booking Status: {prediction}")
