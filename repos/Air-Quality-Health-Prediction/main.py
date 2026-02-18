import pandas as pd
from src.preprocessing import load_and_preprocess
from src.modeling import train_models
from src.visualization import plot_correlation_heatmap
from src.plots import (
    plot_regression_performance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    data = load_and_preprocess('data/AirQualityUCI.csv')
    features = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)', 'T', 'RH']

    reg_model, clf_model, reg_results, clf_results, X_test, y_reg_test, y_pred_reg, y_clf_test, y_clf_pred = train_models(data, features)

    print("Risk_Label distribution in test data:", pd.Series(y_clf_test).value_counts())

    print("\n📈 Regression Results:")
    print(f"R² Score: {reg_results['R2']:.4f}")
    print(f"RMSE: {reg_results['RMSE']:.2f}")

    print("\n🧠 Classification Report:")
    for label, metrics in clf_results.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"Label {label} → Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1-score: {metrics['f1-score']:.2f}")


    disp = ConfusionMatrixDisplay.from_predictions(y_clf_test, y_clf_pred)
    disp.plot()
    plt.title("Classification Confusion Matrix (Sklearn)")
    plt.show()


    plot_correlation_heatmap(data)
    plot_regression_performance(y_reg_test, y_pred_reg)
    if len(set(y_clf_test)) > 1:
        plot_confusion_matrix(y_clf_test, y_clf_pred, labels=[0, 1])


    if len(set(y_clf_test)) > 1:
        y_clf_proba = clf_model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_clf_test, y_clf_proba)

    plot_feature_importance(reg_model, features)

if __name__ == "__main__":
    main()
