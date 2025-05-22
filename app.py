import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

st.title("Electric Vehicle Population Prediction App")

# Extract expected feature names from the model
feature_names = list(model.feature_names_in_)

# Input method selection
input_method = st.radio("Select input method", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    st.subheader("Enter Feature Values Manually")

    # Prepare a dict to hold user inputs
    user_input = {}

    # Dynamically create input fields based on expected features
    for feature in feature_names:
        # Customize input types based on feature name or type knowledge
        if feature in ["County", "City", "State", "Clean Alternative Fuel Vehicle (CAFV) Eligibility", "Electric Utility"]:
            user_input[feature] = st.text_input(feature)
        elif feature in ["Model Year", "Electric Range", "Legislative District"]:
            user_input[feature] = st.number_input(feature, value=0, step=1)
        elif feature == "Base MSRP":
            user_input[feature] = st.number_input(feature, value=0.0)
        else:
            # Default fallback: text input
            user_input[feature] = st.text_input(feature)

    if st.button("Predict"):
        # Build DataFrame for prediction
        input_data = pd.DataFrame([user_input], columns=feature_names)

        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Population: {int(prediction[0])}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.subheader("Upload a CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("Predict from CSV"):
            if list(df.columns) != feature_names:
                st.error(f"CSV columns do not match expected features.\nExpected columns: {feature_names}")
            else:
                try:
                    prediction = model.predict(df)
                    df["Predicted Population"] = prediction
                    st.write("Predictions:")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
