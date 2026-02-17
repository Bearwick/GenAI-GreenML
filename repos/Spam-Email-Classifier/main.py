import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from nltk import PorterStemmer

ps = PorterStemmer()


def convert(word):
    arr = []
    for i in word.split(" "):
        arr.append(ps.stem(i))
    return " ".join(arr)


pd.set_option("display.max_columns", None)
df = pd.read_csv("spam.csv")
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])
df["Message"] = df["Message"].apply(convert)
cv = CountVectorizer(stop_words="english")
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Category"], test_size=0.2)
X_train_count = cv.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_count, y_train)


def is_spam():
    email = input("Enter Email: ")
    org = email
    email = convert(email)
    email = cv.transform([email])
    prediction = model.predict(email)
    if prediction:
        print(f"The email, {org}, is classified as Spam")
    else:
        print(f"The email, {org}, is valid")


is_spam()
















