# SVM Classification on Hospital Data (hospital.csv)

This project applies **Support Vector Machine (SVM)** classification with **Linear** and **RBF** kernels to the `hospital.csv` dataset, which contains hospital diagnostic measurements for predicting a binary medical outcome (`diagnosis`).  

It fulfills **Task 7** requirements: data preparation, model training, hyperparameter tuning, cross-validation, evaluation metrics, and PCA-based decision boundary visualization.

---

## 📂 Project Structure
├── task7_svm_full.py # Full Python script
├── hospital.csv # Hospital dataset file (must be provided)
├── task7_outputs/ # Generated outputs (plots, CSV reports)
│ ├── linear_cv_results.csv
│ ├── rbf_cv_results.csv
│ ├── linear_svm_confusion_matrix.png
│ ├── rbf_svm_confusion_matrix.png
│ ├── linear_pca2_decision_boundary.png
│ ├── rbf_pca2_decision_boundary.png
│ ├── linear_svm_top10_features.csv
│ ├── task7_svm_report.csv


---

## 🧠 Features Implemented
- **Data Preparation**
  - Reads `hospital.csv`
  - Encodes `diagnosis` as binary (`M`=1, `B`=0)
  - Drops unnecessary columns like `id`
  - Ensures all features are numeric
- **Model Training**
  - Linear SVM (`kernel="linear"`)
  - RBF SVM (`kernel="rbf"`)
- **Hyperparameter Tuning**
  - `C` tuning for both models
  - `gamma` tuning for RBF
  - Uses Stratified K-Fold CV and ROC AUC scoring
- **Evaluation**
  - Accuracy, Precision, Recall, F1, ROC AUC
  - Confusion Matrix plots
  - ROC Curves
  - Top 10 feature importance for Linear SVM
- **Visualization**
  - PCA (2D) projection
  - Decision boundary plots for both kernels
- **Outputs**
  - All plots and metrics saved to `/task7_outputs`

---

## 📊 Dataset — `hospital.csv`
This dataset (`hospital.csv`) contains hospital diagnostic measurements for patients, including:
- **Target Column**: `diagnosis` — `M` (Malignant) or `B` (Benign)
- **Features**: Tumor measurements like `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, etc.

> **Note:** `hospital.csv` is **required** to run the project but is **not included** in this repository due to file size/licensing.  
> Please place your `hospital.csv` file in the root folder of the repo before running the script.

---

## ⚙️ Installation
```bash
# Clone this repository
git clone https://github.com/<your-username>/svm-hospital-data-classification.git
cd svm-hospital-data-classification

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

# Install dependencies
pip install pandas numpy matplotlib scikit-learn seaborn
