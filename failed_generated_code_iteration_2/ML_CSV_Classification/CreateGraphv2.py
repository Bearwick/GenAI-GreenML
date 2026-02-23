import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_noise(y, noise_level=0.1):
    """Aggiungi rumore ai dati, tranne al primo e ultimo punto."""
    noise = np.random.normal(0, noise_level, len(y))
    noise[0] = 0
    noise[-1] = 0
    return y + noise

def generate_logarithmic_graph(num_points, final_y_value, noise_level=0.1):
    x = np.linspace(0.1, 10, num_points)  # Evita x=0 per il logaritmo
    y = np.log(x)
    y = (y - y.min()) / (y.max() - y.min())  # Normalizza y tra 0 e 1
    y = y * final_y_value  # Scala a valore finale
    y = add_noise(y, noise_level)  # Aggiungi rumore
    return y

def generate_linear_graph(num_points, final_y_value, noise_level=0.1):
    x = np.linspace(0, 10, num_points)
    y = x
    y = (y - y.min()) / (y.max() - y.min())  # Normalizza y tra 0 e 1
    y = y * final_y_value  # Scala a valore finale
    y = add_noise(y, noise_level)  # Aggiungi rumore
    return y

def create_dataset(num_correct_graphs=100, num_wrong_graphs=100, num_points=50, file_path='grafici.csv', noise_level=0.1):
    X = []
    y = []
    final_y_value = 10  # Punto finale comune per tutti i grafici
    
    # Genera grafici corretti
    for _ in range(num_correct_graphs):
        graph = generate_logarithmic_graph(num_points, final_y_value, noise_level)
        X.append(graph)
        y.append(1)  # Etichetta per grafico corretto
    
    # Genera grafici errati
    for _ in range(num_wrong_graphs):
        graph = generate_linear_graph(num_points, final_y_value, noise_level)
        X.append(graph)
        y.append(0)  # Etichetta per grafico errato
    
    X = np.array(X)
    y = np.array(y)
    
    # Salva in CSV
    df = pd.DataFrame(X)
    df['Label'] = y
    df.to_csv(file_path, index=False)

    # Visualizza un esempio dei grafici generati
    plt.figure(figsize=(12, 8))
    for i in range(len(X)):  # Mostra solo i primi 10 grafici per chiarezza
        plt.plot(X[i], label=f'Grafico {i+1} - {"Corretto" if y[i] == 1 else "Errato"}', alpha=0.5)
    plt.title('Esempi di Grafici Generati')
    plt.xlabel('Tempo')
    plt.ylabel('Valore')
    plt.show()

# Esecuzione del programma
if __name__ == "__main__":
    create_dataset(num_correct_graphs=100, num_wrong_graphs=100, num_points=50, file_path='grafici.csv', noise_level=0.1)
