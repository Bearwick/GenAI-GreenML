"""
Iris Flower Classification using Custom Dataset (CSV)
CodeAlpha Data Science Internship - Task 1
This script performs classification of Iris flower species using a custom CSV dataset.
It includes data loading, cleaning, model training, evaluation, and cross-validation.

"""

# 1. IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 2. LOAD DATASET (CSV)

df = pd.read_csv(r"Iris.csv")

print("Dataset preview:")
print(df.head())

# 3. DATA CLEANING

# Drop Id column if present
if "Id" in df.columns:
    df.drop(columns=["Id"], inplace=True)

# Encode target labels (Species is string)
label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])

# 4. FEATURES & TARGET

X = df.drop("Species", axis=1)
y = df["Species"]

# 5. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. MODEL PIPELINES

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=200))
    ]),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear"))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}


# 7. TRAIN & EVALUATE

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 8. CROSS-VALIDATION

print("\nCross-Validation Results (5-Fold):")
for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}")

print("\nTask-1 completed successfully using CSV dataset!")
