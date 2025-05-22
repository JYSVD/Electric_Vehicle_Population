import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("random_forest.pkl")

model = load_model()

st.title("Electric Vehicle Predictor")

# Input method selection
input_method = st.radio("Select Input Method", ["Manual Input", "Upload CSV"])

# Define the feature names used during training (exclude target 'Vehicle Type')
feature_names = [
    "Model Year", "Make", "Model", "Electric Range", "Base MSRP",
    "Legislative District", "Vehicle Location", "Electric Utility"
    # Add any other features that were used in training
]

# Encode categorical features if needed
def preprocess_input(df):
    # Example: convert categorical to numeric manually or load encoders used during training
    # This is a placeholder — adapt it based on your preprocessing steps
    df_encoded = df.copy()
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    return df_encoded

# ========== Manual Input ==========
if input_method == "Manual Input":
    st.subheader("Enter Feature Values")

    model_year = st.number_input("Model Year", value=2024)
    make = st.text_input("Make", value="Tesla")
    model_name = st.text_input("Model", value="Model S")
    electric_range = st.number_input("Electric Range", value=300)
    base_msrp = st.number_input("Base MSRP", value=79990)
    legislative_district = st.text_input("Legislative District", value="District 1")
    vehicle_location = st.text_input("Vehicle Location", value="Seattle, WA")
    electric_utility = st.text_input("Electric Utility", value="Seattle City Light")

    input_df = pd.DataFrame([[
        model_year, make, model_name, electric_range, base_msrp,
        legislative_district, vehicle_location, electric_utility
    ]], columns=feature_names)

    processed_input = preprocess_input(input_df)

    if st.button("Predict"):
        prediction = model.predict(processed_input)
        st.success(f"Prediction: {prediction[0]}")

# ========== Upload CSV ==========
else:
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        st.write("Input Data:")
        st.dataframe(input_df)

        try:
            processed_input = preprocess_input(input_df)
            prediction = model.predict(processed_input)
            st.write("Predictions:")
            st.write(prediction)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
