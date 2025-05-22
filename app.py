import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

st.title("Electric Vehicle Type Prediction App")

# Extract expected feature names from the model
feature_names = list(model.feature_names_in_)

# Example mappings for categorical features (update as needed)
county_map = {"CountyA": 0, "CountyB": 1}
city_map = {"CityX": 0, "CityY": 1}
state_map = {"State1": 0, "State2": 1}
cafv_map = {"Yes": 1, "No": 0}
utility_map = {"UtilityA": 0, "UtilityB": 1}

def encode_categorical(data):
    for col, mapping in {
        "County": county_map,
        "City": city_map,
        "State": state_map,
        "Clean Alternative Fuel Vehicle (CAFV) Eligibility": cafv_map,
        "Electric Utility": utility_map
    }.items():
        if col in data.columns:
            original_values = data[col].astype(str).unique()
            data[col] = data[col].astype(str).map(mapping)
            unmapped = data[col].isna()
            if unmapped.any():
                st.warning(f"Unmapped values in '{col}': {set(original_values) - set(mapping.keys())}")
            data[col] = data[col].fillna(-1).astype("Int64")
    return data

# Input method selection
input_method = st.radio("Select input method", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    st.subheader("Enter Feature Values Manually")

    user_input = {}
    for feature in feature_names:
        if feature in ["County", "City", "State", "Clean Alternative Fuel Vehicle (CAFV) Eligibility", "Electric Utility"]:
            user_input[feature] = st.text_input(feature)
        elif feature in ["Model Year", "Electric Range", "Legislative District"]:
            user_input[feature] = st.number_input(feature, value=0, step=1)
        elif feature == "Base MSRP":
            user_input[feature] = st.number_input(feature, value=0.0)
        else:
            user_input[feature] = st.text_input(feature, value="Unknown")

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input], columns=feature_names)
        input_df = encode_categorical(input_df)

        if input_df.isna().any().any():
            st.error("Input contains NaN values after encoding. Please check your inputs.")
            st.write("Columns with NaNs:", input_df.columns[input_df.isna().any()].tolist())
        else:
            try:
                prediction = model.predict(input_df)
                st.success(f"Predicted Electric Vehicle Type: {prediction[0]}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

else:
    st.subheader("Upload a CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            st.error(f"The following required columns are missing from the CSV: {missing_cols}")
        else:
            df = df[feature_names]
            df = encode_categorical(df)

            if df.isna().any().any():
                st.error("Data contains NaN values after encoding. Please check your CSV.")
                st.write("Columns with NaNs:", df.columns[df.isna().any()].tolist())
            elif st.button("Predict from CSV"):
                try:
                    prediction = model.predict(df)
                    df["Predicted EV Type"] = prediction
                    st.write("Predictions:")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
