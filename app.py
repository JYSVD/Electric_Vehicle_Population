import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

st.title("Electric Vehicle Population Prediction App")

# Extract expected feature names from the model
feature_names = list(model.feature_names_in_)

# Example mappings for categorical features (replace with your actual categories)
county_map = {"CountyA": 0, "CountyB": 1}
city_map = {"CityX": 0, "CityY": 1}
state_map = {"State1": 0, "State2": 1}
cafv_map = {"Yes": 1, "No": 0}
utility_map = {"UtilityA": 0, "UtilityB": 1}

# Input method selection
input_method = st.radio("Select input method", ["Manual Input", "Upload CSV"])

def encode_categorical(data):
    data["County"] = data["County"].astype(str).map(county_map).fillna(-1).astype("Int64")
    data["City"] = data["City"].astype(str).map(city_map).fillna(-1).astype("Int64")
    data["State"] = data["State"].astype(str).map(state_map).fillna(-1).astype("Int64")
    data["Clean Alternative Fuel Vehicle (CAFV) Eligibility"] = data["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].astype(str).map(cafv_map).fillna(-1).astype("Int64")
    data["Electric Utility"] = data["Electric Utility"].astype(str).map(utility_map).fillna(-1).astype("Int64")
    return data


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
            user_input[feature] = st.text_input(feature)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input], columns=feature_names)

        # Encode categorical columns correctly using .map()
        input_df = encode_categorical(input_df)

        try:
            prediction = model.predict(input_df)
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
                # Encode categorical columns in CSV using .map()
                df = encode_categorical(df)

                try:
                    prediction = model.predict(df)
                    df["Predicted Population"] = prediction
                    st.write("Predictions:")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
