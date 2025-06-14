import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

positive_reviews = ["This movie was fantastic!", "I really enjoyed this film.", "Great acting and storyline.", "A must-see movie!", "I loved every minute of it."] * 10
negative_reviews = ["This movie was terrible.", "I hated this film.", "Poor acting and boring plot.", "Do not waste your time.", "I regret watching this movie."] * 10

reviews = positive_reviews + negative_reviews
sentiments = ['positive'] * 50 + ['negative'] * 50

df = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})

vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

def predict_review_sentiment(model, vectorizer, review):
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]
    return prediction

new_review = "The movie was okay, but not great."
predicted_sentiment = predict_review_sentiment(model, vectorizer, new_review)
print("\nPredicted sentiment for the review:", new_review)
print("Sentiment:", predicted_sentiment)