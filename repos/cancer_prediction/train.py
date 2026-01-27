import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('Cancer_Data.csv')
except FileNotFoundError:
    print("Error: Cancer_Data.csv not found. Make sure the file is in the correct directory.")
    exit()

# Assuming 'diagnosis' column contains 'M' and 'B'
# Convert 'M' to 1 (malignant) and 'B' to 0 (benign) for the target variable
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
y = df['diagnosis']

# Identify top 10 feature columns
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

X = df[feature_names]

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train a Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['benign', 'malignant']))

# 7. Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['benign', 'malignant'], yticklabels=['benign', 'malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 8. Save the trained model
model_filename = 'breast_cancer_model.joblib'
joblib.dump(model, model_filename)
print(f"\nTrained model saved as {model_filename}")

# Save the scaler as well
scaler_filename = 'breast_cancer_scaler.joblib'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved as {scaler_filename}")
