import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv("dataset.csv")
print(data.columns)

data['text'] = data['text'].str.lower()

X = data['text']
y = data['category']

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))

new_text = ["ssr best work"]

new_data = vectorizer.transform(new_text)

result = model.predict(new_data)

print("Prediction:", result)