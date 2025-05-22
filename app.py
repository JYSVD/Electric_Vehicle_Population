import streamlit as st
import pandas as pd
import joblib

# Load the trained model (make sure this model was trained without the lat/lon features)
model = joblib.load("random_forest_model.pkl")

st.title("Electric Vehicle Population Prediction App")

# Input method selection
input_method = st.radio("Select input method", ["Manual Input", "Upload CSV"])

# Features excluding Latitude and Longitude
feature_names = [
    "County", "City", "State", "Model Year",
    "Clean Alternative Fuel Vehicle (CAFV) Eligibility",
    "Electric Range", "Base MSRP", "Legislative District",
    "Electric Utility"
]

# ========== Manual Input ==========
if input_method == "Manual Input":
    st.subheader("Enter Feature Values Manually")

    county = st.text_input("County")
    city = st.text_input("City")
    state = st.text_input("State")
    model_year = st.number_input("Model Year", value=2020, step=1)
    cafv_eligibility = st.text_input("CAFV Eligibility")
    electric_range = st.number_input("Electric Range", value=0, step=1)
    base_msrp = st.number_input("Base MSRP", value=0.0)
    district = st.number_input("Legislative District", value=0, step=1)
    electric_utility = st.text_input("Electric Utility")

    input_data = pd.DataFrame([[
        county, city, state, model_year,
        cafv_eligibility, electric_range, base_msrp,
        district, electric_utility
    ]], columns=feature_names)

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Population: {int(prediction[0])}")

# ========== CSV Upload ==========
else:
    st.subheader("Upload a CSV File")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("Predict from CSV"):
            # Make sure the CSV contains the correct features matching the model's expected input
            prediction = model.predict(df)
            df["Predicted Population"] = prediction
            st.write("Predictions:")
            st.dataframe(df)
