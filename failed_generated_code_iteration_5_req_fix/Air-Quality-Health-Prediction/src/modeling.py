from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report

def train_models(data, features):
    X = data[features]
    y_reg = data['Hospital_Visits']
    y_clf = data['Risk_Label']

    # Split data
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=0)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=0)

    # Train regression model
    reg_model = RandomForestRegressor().fit(X_train, y_reg_train)
    y_pred_reg = reg_model.predict(X_test)

    # Train classification model
    clf_model = RandomForestClassifier().fit(X_train, y_clf_train)
    y_clf_pred = clf_model.predict(X_test)

    # Evaluation metrics
    regression_results = {
        'R2': r2_score(y_reg_test, y_pred_reg),
        'RMSE': mean_squared_error(y_reg_test, y_pred_reg) ** 0.5
    }

    classification_results = classification_report(y_clf_test, y_clf_pred, output_dict=True)

    return (
        reg_model, clf_model, regression_results, classification_results,
        X_test, y_reg_test, y_pred_reg, y_clf_test, y_clf_pred
    )
