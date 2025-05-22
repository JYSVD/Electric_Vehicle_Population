import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Load saved model, feature names, and encoders
MODEL_FILENAME = "Random_Forest_model.pkl"  # replace with your actual file path

@st.cache_data
def load_model():
    model, feature_names, label_encoders, target_le = joblib.load(MODEL_FILENAME)
    return model, feature_names, label_encoders, target_le

model, feature_names, label_encoders, target_le = load_model()

st.title("Electric Vehicle Type Classifier")

st.write("Upload CSV or enter input manually to predict EV type.")

# Option to upload CSV
uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])

def preprocess_input(df):
    # Apply label encoding on categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    # Fill missing columns with 0 or mean (depending on your preference)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[feature_names]
    return df

if uploaded_file is not None:
    # Read CSV
    input_df = pd.read_csv(uploaded_file)
    input_df_processed = preprocess_input(input_df)
    
    preds = model.predict(input_df_processed)
    pred_labels = target_le.inverse_transform(preds)
    
    st.write("### Predictions")
    results = input_df.copy()
    results['Predicted EV Type'] = pred_labels
    st.dataframe(results)

else:
    # Manual input form
    st.write("### Enter feature values manually")

    input_dict = {}
    for feature in feature_names:
        val = st.text_input(feature)
        input_dict[feature] = val if val != '' else '0'  # default 0 if empty

    if st.button("Predict"):
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Convert columns to numeric where possible
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
        # Preprocess
        input_df_processed = preprocess_input(input_df)
        
        # Predict
        pred = model.predict(input_df_processed)[0]
        pred_label = target_le.inverse_transform([pred])[0]
        
        st.write(f"### Predicted Electric Vehicle Type: **{pred_label}**")

# Optional: Display performance metrics summary (hardcoded or loaded from file)
# For demo, let's just show a confusion matrix if you have test data loaded

if st.checkbox("Show example confusion matrix"):
    # This example uses random data; replace with your actual test results
    y_true = [0, 1, 0, 2, 1, 0]  # example true labels
    y_pred = [0, 0, 0, 2, 1, 1]  # example predicted labels
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
