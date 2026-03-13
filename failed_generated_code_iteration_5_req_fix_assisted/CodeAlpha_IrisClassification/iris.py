import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
try:
    data = pd.read_csv("C:/Users/Lalit Pathak/python2.0/Iris.csv")
except FileNotFoundError:
    print("Error: Dataset not found. Please check the file path.")
    exit()

# 2. Quick Look at Data
print("First 5 rows of dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Drop ID column if exists
if 'Id' in data.columns:
    data.drop(columns=['Id'], inplace=True)

# 3. Features (X) and Labels (y)
if 'Species' not in data.columns:
    print("Error: 'Species' column not found in dataset.")
    exit()

X = data.drop(columns=['Species'])
y = data['Species']

# Encode labels (Species -> numeric)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Model Training - Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Prediction
y_pred_log = log_reg.predict(X_test)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log, target_names=le.classes_))

# 6. Model Training - Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prediction
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# 7. Confusion Matrix Visualization (Random Forest)
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()