import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def load_model(file_path='model.pkl'):
    data = joblib.load(file_path)
    return data['model'], data['scaler']

def preprocess_data(X, scaler):
    X_scaled = scaler.transform(X)
    return X_scaled

def classify_new_trends(model, scaler, new_data):
    X_new_scaled = preprocess_data(new_data, scaler)
    predictions = model.predict(X_new_scaled)
    return predictions

def plot_classified_trends(X, predictions, output_image='new_classified_trends.png'):
    plt.figure(figsize=(15, 10))
    
    for i, (trend, prediction) in enumerate(zip(X, predictions)):
        color = 'green' if prediction == 1 else 'red'
        plt.plot(trend, color=color, alpha=0.7, label=f'Trend {i+1}: {"Corretto" if prediction == 1 else "Errato"}')
    
    plt.title('Tutti i Grafici Classificati')
    plt.xlabel('Tempo')
    plt.ylabel('Valore')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.show()

def load_and_prepare_data(file_path):
    try:
        data = pd.read_csv(file_path, header=None).values
        if data.shape[1] != 50:  # Assuming the model expects 50 features
            raise ValueError(f"Expected 50 features, but got {data.shape[1]}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    model, scaler = load_model('model.pkl')
    new_data = load_and_prepare_data('new_trends.csv')
    
    if new_data is not None:
        predictions = classify_new_trends(model, scaler, new_data)
        
        for i, prediction in enumerate(predictions):
            print(f"Trend {i+1} is classified as: {'Correct' if prediction == 1 else 'Incorrect'}")
        
        plot_classified_trends(new_data, predictions)
    else:
        print("Could not proceed due to data loading error.")