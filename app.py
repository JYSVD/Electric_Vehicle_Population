import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

st.title("Electric Vehicle Type Prediction App")

# Extract expected feature names from the model
feature_names = list(model.feature_names_in_)

# Label mapping (you should confirm this with your model!)
ev_type_map = {
    0: "Battery Electric Vehicle (BEV)",
    1: "Plug-in Hybrid Electric Vehicle (PHEV)"
}

# Update with the actual categories in your training data
county_map = {"Yakima": 0, "Kitsap": 1}
city_map = {"Yakima": 0, "Kingston": 1}
state_map = {"WA": 0, "WA": 1}
cafv_map = {"Not": 1, "Clean": 0}
utility_map = {"PACIFICORP": 0, "PUGET SOUND ENERGY INC": 1}

def encode_categorical(data):
    mappings = {
        "County": county_map,
        "City": city_map,
        "State": state_map,
        "Clean Alternative Fuel Vehicle (CAFV) Eligibility": cafv_map,
        "Electric Utility": utility_map
    }

    for col, mapping in mappings.items():
        if col in data.columns:
            original_values = data[col].astype(str).unique()
            unmapped = set(original_values) - set(mapping.keys())
            if unmapped:
                st.warning(f"⚠️ Unmapped values in '{col}': {unmapped}")
            data[col] = data[col].astype(str).map(mapping).fillna(-1).astype("Int64")

    return data

# Input method
input_method = st.radio("Select input method", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    st.subheader("Enter Feature Values Manually")

    user_input = {}
    for feature in feature_names:
        if feature in ["County", "City", "State", "Clean Alternative Fuel Vehicle (CAFV) Eligibility", "Electric Utility"]:
            user_input[feature] = st.text_input(f"{feature}")
        elif feature in ["Model Year", "Electric Range", "Legislative District"]:
            user_input[feature] = st.number_input(feature, value=2020, step=1)
        elif feature == "Base MSRP":
            user_input[feature] = st.number_input(feature, value=30000.0)
        else:
            user_input[feature] = st.text_input(feature, value="Unknown")

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input], columns=feature_names)
        input_df = encode_categorical(input_df)

        try:
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)
            label = ev_type_map.get(prediction[0], "Unknown")
            st.success(f"Predicted Electric Vehicle Type: {label}")
            st.info(f"Prediction Confidence - BEV: {proba[0][0]:.2f}, PHEV: {proba[0][1]:.2f}")
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

            if st.button("Predict from CSV"):
                try:
                    prediction = model.predict(df)
                    proba = model.predict_proba(df)
                    df["Predicted EV Type"] = [ev_type_map.get(p, "Unknown") for p in prediction]
                    df["BEV Confidence"] = proba[:, 0]
                    df["PHEV Confidence"] = proba[:, 1]
                    st.write("Predictions:")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
