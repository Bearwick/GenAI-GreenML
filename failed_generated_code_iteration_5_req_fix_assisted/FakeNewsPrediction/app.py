from flask import Flask, request, render_template
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    result = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"
    return render_template("index.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
