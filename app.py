import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    # Adjust filename to your model file
    model_data = joblib.load("random_forest_model.pkl")
    # Unpack contents (adjust according to what you saved)
    model = model_data['model']
    feature_names = model_data['feature_names']
    label_encoders = model_data.get('label_encoders', {})  # may be empty dict
    target_le = model_data.get('target_le', None)          # may be None if not used
    return model, feature_names, label_encoders, target_le

model, feature_names, label_encoders, target_le = load_model()

st.title("Electric Vehicle Type Prediction")

input_method = st.radio("Select input method", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    inputs = {}
    for feature in feature_names:
        if feature in label_encoders:
            options = label_encoders[feature].classes_
            inputs[feature] = st.selectbox(f"Select {feature}", options)
        else:
            inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([inputs])
        # Encode categorical inputs
        for col, le in label_encoders.items():
            input_df[col] = le.transform(input_df[col])
        prediction = model.predict(input_df)
        if target_le:
            prediction_label = target_le.inverse_transform(prediction)[0]
        else:
            prediction_label = prediction[0]
        st.success(f"Predicted Vehicle Type: {prediction_label}")

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        st.write("Data preview:")
        st.dataframe(df)
        if st.button("Predict from CSV"):
            predictions = model.predict(df[feature_names])
            if target_le:
                predictions = target_le.inverse_transform(predictions)
            df["Predicted Vehicle Type"] = predictions
            st.success("Predictions done:")
            st.dataframe(df)
