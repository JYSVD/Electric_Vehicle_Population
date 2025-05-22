import streamlit as st
import pandas as pd
import joblib

# Load model and related metadata
@st.cache_resource
def load_model():
    return joblib.load("electric_vehicle_model.pkl")

model, feature_names, label_encoders, target_le = load_model()

st.title("Electric Vehicle Type Prediction")

st.subheader("Input EV Information")

# Manual inputs
make = st.selectbox("Make", label_encoders["Make"].classes_)
model_name = st.text_input("Model")
model_year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2020)
electric_range = st.number_input("Electric Range (miles)", min_value=0, value=100)
base_msrp = st.number_input("Base MSRP ($)", min_value=0, value=30000)
legislative_district = st.selectbox("Legislative District", label_encoders["Legislative District"].classes_)
electric_utility = st.selectbox("Electric Utility", label_encoders["Electric Utility"].classes_)
cafe_vehicle_description = st.text_input("CAFE Vehicle Description")
vehicle_location = st.text_input("Vehicle Location")
county = st.selectbox("County", label_encoders["County"].classes_)

# Encode categorical fields
input_data = pd.DataFrame([[
    label_encoders["Make"].transform([make])[0],
    model_name,  # You can skip this if not used by model
    model_year,
    electric_range,
    base_msrp,
    label_encoders["Legislative District"].transform([legislative_district])[0],
    label_encoders["Electric Utility"].transform([electric_utility])[0],
    cafe_vehicle_description,  # Skip if unused by model
    vehicle_location,          # Skip if unused by model
    label_encoders["County"].transform([county])[0]
]], columns=feature_names)

# Predict
if st.button("Predict Vehicle Type"):
    prediction = model.predict(input_data)
    predicted_label = target_le.inverse_transform(prediction)[0]
    st.success(f"Predicted Vehicle Type: **{predicted_label}**")
