import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

def plot_regression_performance(y_true, y_pred):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title("Regression: Actual vs Predicted Hospital Visits")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap='Blues'
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1]

    plt.figure(figsize=(8, 5))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(len(feature_names)), importances[sorted_indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in sorted_indices], rotation=45)
    plt.tight_layout()
    plt.show()

