import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("✅ Script started")  # DEBUG line

def train_model():
    print("📥 Loading datasets...")  # DEBUG
    df_fake = pd.read_csv("Fake.csv")
    df_real = pd.read_csv("True.csv")

    print("🏷️  Labeling data...")  # DEBUG
    df_fake["label"] = 0
    df_real["label"] = 1

    df = pd.concat([df_fake, df_real])
    df = df[['text', 'label']]

    print("🔀 Splitting data...")  # DEBUG
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

    print("🧠 Training model...")  # DEBUG
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    print("💾 Saving model...")  # DEBUG
    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    print("✅ Model training complete and saved.")

if __name__ == '__main__':
    print("🚀 Starting training process...")  # DEBUG
    train_model()
