import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Graph data
    y = data.iloc[:, -1].values   # Labels (0 for incorrect, 1 for correct)
    return X, y

# 2. Preprocess and extract features
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# 3. Create and train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

# 4. Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.2f}")
    return y_pred

# 5. Save and load model
def save_model(model, scaler, file_path='model.pkl'):
    joblib.dump({'model': model, 'scaler': scaler}, file_path)

def load_model(file_path='model.pkl'):
    data = joblib.load(file_path)
    return data['model'], data['scaler']

# 6. Visualize results
def plot_classified_graphs(X, y_true, y_pred, output_image='classified_graphs.png'):
    plt.figure(figsize=(12, 8))

    colors = {
        'true_correct': 'green',
        'true_incorrect': 'red',
        'pred_correct': 'blue',
        'pred_incorrect': 'orange'
    }

    for i in range(len(y_true)):
        true_color = colors['true_correct'] if y_true[i] == 1 else colors['true_incorrect']
        pred_color = colors['pred_correct'] if y_pred[i] == 1 else colors['pred_incorrect']

        plt.plot(X[i], color=true_color, alpha=0.5)
        plt.plot(X[i], color=pred_color, linestyle='--', alpha=0.5)

    plt.title('Graphs with Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors['true_correct'], lw=2),
                    Line2D([0], [0], color=colors['true_incorrect'], lw=2),
                    Line2D([0], [0], color=colors['pred_correct'], lw=2, linestyle='--'),
                    Line2D([0], [0], color=colors['pred_incorrect'], lw=2, linestyle='--')]

    plt.legend(custom_lines, ['True Correct', 'True Incorrect', 'Predicted Correct', 'Predicted Incorrect'])
    
    plt.savefig(output_image)
    plt.show()

# 7. Cross-validation
def cross_validate_model(X, y):
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

# Main execution
if __name__ == "__main__":
    # Load data
    file_path = 'grafici.csv'  # Replace with your CSV file path
    X, y = load_data(file_path)
    X_scaled, scaler = preprocess_data(X)

    # Perform cross-validation
    cross_validate_model(X_scaled, y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, scaler)

    # Visualize results
    plot_classified_graphs(X_test, y_test, y_pred, output_image='classified_graphs.png')

    # Test on a single correct graph
    correct_graph = X_test[y_test == 1][0].reshape(1, -1)  # Get first correct graph from test set
    prediction = model.predict(correct_graph)
    print(f"Prediction for a correct graph: {'Correct' if prediction[0] == 1 else 'Incorrect'}")

    # Feature importance
    feature_importance = model.feature_importances_
    for i, importance in enumerate(feature_importance):
        print(f"Feature {i+1} importance: {importance:.4f}")