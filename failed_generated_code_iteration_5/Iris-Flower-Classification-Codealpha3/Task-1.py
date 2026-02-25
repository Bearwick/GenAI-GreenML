"""
Iris Flower Classification using Custom Dataset (CSV)
CodeAlpha Data Science Internship - Task 1
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r"Iris.csv")

if "Id" in df.columns:
    df.drop(columns=["Id"], inplace=True)

label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])

X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Model: Random Forest")
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Mean Accuracy: {scores.mean():.4f}")