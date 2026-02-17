import os
import email
from email import policy
import re
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def extract_email_content(filepath):
    """Extract sender, subject, body, and URLs from an .eml file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_file(f, policy=policy.default)
        
        sender = msg['From'] or ''
        subject = msg['Subject'] or ''
        
        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif part.get_content_type() == 'text/html':
                    html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(html, 'html.parser')
                    body += soup.get_text()
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', body)
        
        return {
            'sender': sender,
            'subject': subject,
            'body': body,
            'urls': urls
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def load_emails(spam_dir, good_dir):
    """Load .eml files from spam and good directories."""
    data = []
    for filename in os.listdir(spam_dir):
        if filename.endswith('.eml'):
            filepath = os.path.join(spam_dir, filename)
            email_data = extract_email_content(filepath)
            if email_data:
                email_data['filename'] = filename
                email_data['label'] = 'spam'
                data.append(email_data)
    
    for filename in os.listdir(good_dir):
        if filename.endswith('.eml'):
            filepath = os.path.join(good_dir, filename)
            email_data = extract_email_content(filepath)
            if email_data:
                email_data['filename'] = filename
                email_data['label'] = 'good'
                data.append(email_data)
    
    return pd.DataFrame(data)

def train_spam_classifier(df):
    """Train a spam detection model."""
    df['text'] = df['subject'] + ' ' + df['body']
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model, vectorizer

if __name__ == "__main__":
    spam_dir = "emails/spam"
    good_dir = "emails/good"
    
    print("Loading emails...")
    df = load_emails(spam_dir, good_dir)
    
    print("Training spam classifier...")
    model, vectorizer = train_spam_classifier(df)
    
    print("Predicting for a sample email...")
    new_email = df.iloc[0]['text']
    new_email_features = vectorizer.transform([new_email])
    prediction = model.predict(new_email_features)
    print(f"Prediction for {df.iloc[0]['filename']}: {prediction[0]}")
    
    df.to_csv("email_analysis.csv", index=False)
    print("Results saved to email_analysis.csv")
