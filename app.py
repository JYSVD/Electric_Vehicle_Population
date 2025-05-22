import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and label encoder
MODEL_FILENAME = "Random_Forest_model.pkl"  # Change this to match your saved model filename
model = joblib.load(MODEL_FILENAME)

st.set_page_config(page_title="EV Type Classifier", layout="centered")
st.title("🚗 Electric Vehicle Type Classifier")

st.markdown("""
Upload a CSV file or enter EV information manually to predict the **Electric Vehicle Type**.
""")

# Sample input fields (adjust to match your dataset)
input_features = [
    'VIN (1-10)', 'County', 'City', 'State', 'Postal Code', 'Model Year',
    'Make', 'Model', 'Electric Range', 'Base MSRP', 'Legislative District',
    'DOL Vehicle ID', 'Vehicle Location', 'Electric Utility', '2020 Census Tract'
]

# Optional: Provide mapping or label encoding
def preprocess_input(df):
    # Basic numeric fill (you can extend this to full encoding you used during training)
    df = df.copy()
    df.fillna(0, inplace=True)

    # Drop any target column if included
    if 'Electric Vehicle Type' in df.columns:
        df = df.drop(columns=['Electric Vehicle Type'])

    return df

# CSV upload
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])
if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(user_data.head())

    input_data = preprocess_input(user_data)
    prediction = model.predict(input_data)

    st.subheader("🔮 Predictions")
    st.write(prediction)

else:
    st.subheader("✍️ Manual Input")
    user_input = {}
    for feature in input_features:
        value = st.text_input(f"{feature}", "")
        user_input[feature] = value

    if st.button("Predict EV Type"):
        input_df = pd.DataFrame([user_input])
        input_df = preprocess_input(input_df)
        prediction = model.predict(input_df)

        st.success(f"Predicted EV Type: {prediction[0]}")

# Optional: Display confusion matrix (from training)
if st.checkbox("Show Model Performance Summary (Confusion Matrix)"):
    try:
        # Assuming you saved confusion matrix image from training phase
        st.image("conf_matrices/Random_Forest_cm.png", caption="Confusion Matrix")
    except:
        st.warning("Confusion matrix image not found.")
