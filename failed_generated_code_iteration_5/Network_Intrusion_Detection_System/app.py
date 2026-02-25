import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from src.preprocessing import preprocess_data

st.title("🛡️ Network Intrusion Detection")

# Load trained artifacts
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
model = joblib.load("models/xgb_model.pkl")
train_columns = joblib.load("models/train_columns.pkl")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.head())

    # Preprocess
    X_scaled, y_true = preprocess_data(df, fit=False, scaler=scaler, columns=train_columns)
    X_pca = pca.transform(X_scaled)

    # Predict
    preds = model.predict(X_pca)
    df["Prediction"] = ["Normal" if p==0 else "Attack" for p in preds]
    st.write("Predictions:")
    st.dataframe(df)

    # Metrics if labels exist
    if y_true is not None:
        st.write("📊 Classification Report:")
        report = classification_report(y_true, preds, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

        # ROC
        y_probs = model.predict_proba(X_pca)[:,1]
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        st.write(f"ROC-AUC Score: {roc_auc:.3f}")
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())
