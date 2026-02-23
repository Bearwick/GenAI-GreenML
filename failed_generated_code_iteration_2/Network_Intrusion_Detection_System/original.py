import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_curve, auc
from xgboost import XGBClassifier
from src.preprocessing import preprocess_data

# Load raw datasets
train_df = pd.read_csv("data/raw/Train.txt", header=None)
test_df = pd.read_csv("data/raw/Test.txt", header=None)
print("✅ Data loaded")

# Preprocessing
X_train_scaled, y_train, scaler, train_columns = preprocess_data(train_df, fit=True)
X_test_scaled, y_test = preprocess_data(test_df, fit=False, scaler=scaler, columns=train_columns)
print("✅ Preprocessing done")

# Save training columns
joblib.dump(train_columns, "models/train_columns.pkl")

# PCA
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"✅ PCA reduced features to {X_train_pca.shape[1]}")

# XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train_pca, y_train)
print("✅ Model trained")

# Predictions & metrics
y_pred = model.predict(X_test_pca)
y_probs = model.predict_proba(X_test_pca)[:, 1]
print("\n📊 Classification Report")
print(classification_report(y_test, y_pred))

# ROC-AUC
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
print(f"ROC-AUC Score: {roc_auc:.3f}")

# Save artifacts
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(model, "models/xgb_model.pkl")
print("✅ Models saved")
