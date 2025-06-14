import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# Expanded synthetic dataset
positive_feedback = [
    "I'm absolutely thrilled with this purchase!",
    "The product exceeded my expectations.",
    "Incredible value for the money.",
    "The best [product type] I've ever used.",
    "I would highly recommend this to anyone.",
    "A fantastic product that delivers on its promises.",
    "This has made my life so much easier.",
    "I can't stop raving about this product!",
    "The quality is outstanding.",
    "A truly exceptional product.",
    "Works perfectly and looks great!",
    "So glad I decided to buy this.",
    "This is exactly what I was looking for.",
    "Excellent customer service and a great product.",
    "I'm a very happy customer!",
    " exceeded my expectations",
    "great value",
    "highly recommend",
    "fantastic",
    "love it"
] * 5

negative_feedback = [
    "I'm extremely disappointed with this product.",
    "The product was defective upon arrival.",
    "A complete waste of money.",
    "The worst [product type] I've ever purchased.",
    "I would not recommend this to anyone.",
    "A terrible product that doesn't work as advertised.",
    "This has made my life more difficult.",
    "I regret buying this product.",
    "The quality is subpar.",
    "A truly awful product.",
    "Doesn't work at all and looks cheap!",
    "So sad I decided to buy this.",
    "This is not what I was looking for.",
    "Terrible customer service and a bad product.",
    "I'm a very unhappy customer!",
    "did not meet my expectations",
    "waste of money",
    "do not recommend",
    "awful",
    "hate it"
] * 5

texts = positive_feedback + negative_feedback
labels = ['good'] * len(positive_feedback) + ['bad'] * len(negative_feedback)

df = pd.DataFrame({'text': texts, 'label': labels})

vectorizer = TfidfVectorizer(max_features=300, stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred, pos_label='good')
recall = recall_score(y_test, y_pred, pos_label='good')
f1 = f1_score(y_test, y_pred, pos_label='good')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

def text_preprocess_vectorize(texts, vectorizer):
    X = vectorizer.transform(texts)
    return X

new_texts = ["This is an amazing product!", "The worst product ever."]
new_X = text_preprocess_vectorize(new_texts, vectorizer)
print("\nVectorized feature matrix for new texts:")
print(new_X.toarray())