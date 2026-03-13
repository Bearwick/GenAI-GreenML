
# 🛡️ Network Intrusion Detection System (NIDS)

## Overview

This project is a **Machine Learning-powered web application** to detect network intrusions using the **NSL-KDD dataset**. The system classifies network traffic as either **Normal** or **Attack**, helping organizations proactively identify and mitigate potential cybersecurity threats.

The model leverages **Principal Component Analysis (PCA)** for dimensionality reduction and **XGBoost** for high-performance classification.

---

## 🔹 Features

* **Web-based interface using Streamlit** for easy file upload and predictions.
* **Automatic preprocessing**:

  * Handles numeric and categorical features.
  * Aligns uploaded files to the training features automatically.
  * Supports files with or without labels.
* **Dimensionality reduction** with PCA to reduce computational overhead while preserving 95% variance.
* **XGBoost classification** for accurate detection of attacks.
* **Evaluation metrics**:

  * Precision, Recall, F1-score
  * Accuracy
  * ROC-AUC and ROC Curve visualization
* **Prediction for new, unseen network traffic**.



## 🔹 Technology Stack

| Component         | Tool / Library                |
| ----------------- | ----------------------------- |
| Backend / ML      | Python, scikit-learn, XGBoost |
| Frontend          | Streamlit                     |
| Data Handling     | pandas, numpy                 |
| Model Persistence | joblib                        |
| Visualization     | matplotlib, seaborn           |




Network_Intrusion_System/
│
├─ data/
│   └─ raw/         # NSL-KDD train & test files
├─ models/          # Saved scaler, PCA, and XGBoost model
├─ src/
│   └─ preprocessing.py
├─ train.py         # Training pipeline
├─ app.py           # Streamlit web app
└─ README.md


## 🔹 Usage

### 1. Training the Model

```
python train.py
```

* Loads the NSL-KDD dataset.
* Preprocesses features.
* Applies PCA (95% variance).
* Trains XGBoost classifier.
* Saves trained models: `scaler.pkl`, `pca.pkl`, `xgb_model.pkl`, `train_columns.pkl`.

---

### 2. Running the Web App

```
streamlit run app.py
```

* Upload a CSV file containing **network traffic features**.
* The app will preprocess the file, apply PCA, and predict whether each row is **Normal** or **Attack**.
* If the CSV contains labels (`attack_type`), evaluation metrics and ROC curve are displayed.
* If labels are missing, only predictions are shown.

---

### 3. Input CSV Format

* Must include numeric and categorical features used in training.
* Example (first few columns):

```
duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,count,srv_count, ...
```

* Label column (`attack_type`) is optional.

---

## 🔹 Sample Output

| duration | protocol_type | service | flag | src_bytes | ... | Prediction |
| -------- | ------------- | ------- | ---- | --------- | --- | ---------- |
| 0        | tcp           | http    | SF   | 181       | ... | Normal     |
| 0        | tcp           | http    | SF   | 239       | ... | Attack     |

* Classification Report:

```
              precision    recall    f1-score     support
Normal          0.68        0.94      0.79         9711
Attack          0.93        0.66      0.77         12833
accuracy                              0.78         22544
```

* ROC-AUC Score: 0.889

---

## 🔹 Key Advantages

* Handles **high-dimensional network data** efficiently using PCA.
* High classification accuracy for network attacks using XGBoost.
* Flexible input: works with **partial features**, optional labels.
* Web-based interface makes it **easy for non-technical users** to detect intrusions.

---

## 🔹 Future Enhancements

* Support **real-time streaming network traffic** for live intrusion detection.
* Expand to **multi-class classification** for detailed attack types.
* Add **visual analytics dashboard** for better monitoring.

---

## 🔹 References

1. [NSL-KDD Dataset](https://github.com/defcom17/NSL_KDD)
2. [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
3. [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
4. [Streamlit](https://streamlit.io/)



