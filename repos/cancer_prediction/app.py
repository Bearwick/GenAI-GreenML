import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load('breast_cancer_model.joblib')
except FileNotFoundError:
    st.error("Error: Trained model file 'breast_cancer_model.joblib' not found. Make sure it's in the same directory as this app.")
    st.stop()

# Load the saved scaler
try:
    scaler = joblib.load('breast_cancer_scaler.joblib')
except FileNotFoundError:
    st.error("Error: Scaler file 'breast_cancer_scaler.joblib' not found. Make sure it's in the same directory as this app.")
    st.stop()

# Define the feature names that your model expects.
# **IMPORTANT:** This list MUST be in the same order as the features during training
feature_names = [
    "texture_worst",
    "radius_se",
    "symmetry_worst",
    "concave points_mean",
    "area_se",
    "area_worst",
    "radius_worst",
    "concave points_worst",
    "concavity_mean",
    "fractal_dimension_se"
]

st.title("Breast Cancer Prediction App")
st.write("Enter the tumor characteristics to get a prediction.")

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    input_value = st.number_input(f"Enter {feature}")
    input_data[feature] = input_value

# Create a button to trigger prediction
if st.button("Predict"):
    # Convert input data to a Pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # Make the prediction
    try:
        prediction = model.predict(input_scaled)[0]

        # Display the prediction
        st.subheader("Prediction:")
        if prediction == 1:
            st.markdown("<p style='color:red; font-size:24px;'>Malignant</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:green; font-size:24px;'>Benign</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
