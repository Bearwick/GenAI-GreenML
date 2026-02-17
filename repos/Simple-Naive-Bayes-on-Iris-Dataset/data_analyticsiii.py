import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

iris = pd.read_csv("iris.csv")

label_encoder = LabelEncoder()
iris["Species"] = label_encoder.fit_transform(iris["Species"])

X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_train, X_test, y_train, y_test, title):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    print(f"{title}:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Error Rate: {error_rate:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}\n")
    
    return cm


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

gaussian_cm = evaluate_model(GaussianNB(), X_train, X_test, y_train, y_test, "Gaussian Naïve Bayes")
plot_confusion_matrix(gaussian_cm, "Gaussian Naïve Bayes Confusion Matrix")

multinomial_cm = evaluate_model(MultinomialNB(), X_train, X_test, y_train, y_test, "Multinomial Naïve Bayes")
plot_confusion_matrix(multinomial_cm, "Multinomial Naïve Bayes Confusion Matrix")

bernoulli_cm = evaluate_model(BernoulliNB(), X_train, X_test, y_train, y_test, "Bernoulli Naïve Bayes")
plot_confusion_matrix(bernoulli_cm, "Bernoulli Naïve Bayes Confusion Matrix")
