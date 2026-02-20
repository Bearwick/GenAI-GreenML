import joblib
import pandas as pd

# Load the trained model
try:
    model = joblib.load('breast_cancer_model.joblib')
except FileNotFoundError:
    print("Error: Trained model file 'breast_cancer_model.joblib' not found.")
    exit()

# Load the saved scaler
try:
    scaler = joblib.load('breast_cancer_scaler.joblib')
except FileNotFoundError:
    print("Error: Scaler file 'breast_cancer_scaler.joblib' not found.")
    exit()

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

print("Enter the following tumor characteristics:")

input_features = {}
for feature in feature_names:
    while True:
        try:
            value = float(input(f"Enter value for {feature}: "))
            input_features[feature] = value
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Convert input features to a Pandas DataFrame
input_df = pd.DataFrame([input_features])

# Scale the input data using the loaded scaler
input_scaled = scaler.transform(input_df)

# Make the prediction
try:
    prediction = model.predict(input_scaled)[0]

    print("\nPrediction:")
    if prediction == 1:
        print("Malignant")
    else:
        print("Benign")

except Exception as e:
    print(f"An error occurred during prediction: {e}")