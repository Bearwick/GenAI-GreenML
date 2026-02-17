# Task 7 — SVM with Linear & RBF kernels on hospital.csv

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 2. Load dataset
df = pd.read_csv("hospital.csv")
print("Dataset shape:", df.shape)

# 3. Prepare data
# Target: diagnosis (M = Malignant, B = Benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop(columns=['diagnosis', 'id'], errors='ignore')
y = df['diagnosis']

# 4. Train/Test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Build SVM pipelines (Scaling + SVM)
linear_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="linear", probability=True, random_state=42))
])

rbf_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", probability=True, random_state=42))
])

# 6. Hyperparameter grids (small for quick run)
lin_params = {"svc__C": [0.1, 1, 10]}
rbf_params = {"svc__C": [1, 10], "svc__gamma": ["scale", 0.01, 0.1]}

# 7. Cross-validation setup
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# 8. Grid search for both models
lin_grid = GridSearchCV(linear_svm, lin_params, scoring="roc_auc", cv=cv)
rbf_grid = GridSearchCV(rbf_svm, rbf_params, scoring="roc_auc", cv=cv)

lin_grid.fit(X_train, y_train)
rbf_grid.fit(X_train, y_train)

print("Best Linear SVM params:", lin_grid.best_params_)
print("Best RBF SVM params:", rbf_grid.best_params_)

# 9. Evaluation function
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Results:")
    print(" Accuracy :", accuracy_score(y_test, y_pred))
    print(" Precision:", precision_score(y_test, y_pred))
    print(" Recall   :", recall_score(y_test, y_pred))
    print(" F1 Score :", f1_score(y_test, y_pred))
    print(" ROC AUC  :", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{name} ROC Curve")
    plt.show()

# 10. Evaluate both models
evaluate(lin_grid.best_estimator_, X_test, y_test, "Linear SVM")
evaluate(rbf_grid.best_estimator_, X_test, y_test, "RBF SVM")

# 11. Feature importance (only for linear SVM)
lin_coef = lin_grid.best_estimator_.named_steps["svc"].coef_.ravel()
feat_importance = pd.Series(abs(lin_coef), index=X.columns).sort_values(ascending=False)
print("\nTop 10 Features (Linear SVM):\n", feat_importance.head(10))

# 12. PCA for 2D decision boundary visualization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

# Train SVMs on 2D PCA data for plotting
lin_2d = SVC(kernel="linear", C=1, probability=True).fit(X_pca, y)
rbf_2d = SVC(kernel="rbf", C=1, gamma="scale", probability=True).fit(X_pca, y)

def plot_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.show()
    

plot_boundary(lin_2d, X_pca, y, "Linear SVM Decision Boundary (PCA 2D)")
plot_boundary(rbf_2d, X_pca, y, "RBF SVM Decision Boundary (PCA 2D)")

# ...existing code...

# Get predictions and probabilities from best RBF SVM
y_pred = rbf_grid.best_estimator_.predict(X_test)
y_proba = rbf_grid.best_estimator_.predict_proba(X_test)[:, 1]

# Create DataFrame for cleaned/scaled test features
X_test_scaled = rbf_grid.best_estimator_.named_steps["scaler"].transform(X_test)
X_test_cleaned = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Add actual, predicted, and probability columns
X_test_cleaned['Actual'] = y_test.values
X_test_cleaned['Predicted'] = y_pred
X_test_cleaned['Probability'] = y_proba


# Function to plot a nice confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Benign", "Malignant"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


# Save to CSV
X_test_cleaned.to_csv(r"D:\AIML PROJECTS\hospital_test_results.csv", index=False)

print("File saved as hospital_test_results.csv")

# ...existing code...