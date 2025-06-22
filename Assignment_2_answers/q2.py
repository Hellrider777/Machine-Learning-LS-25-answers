import pandas as pd
import numpy as np
import re
import string
import nltk
import gensim.downloader as api
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import contractions
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv("tweets.csv")
df = df[['airline_sentiment', 'text']].dropna()


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = contractions.fix(text)  # Expand contractions
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtag symbol only (keep word)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return tokens

df['tokens'] = df['text'].apply(preprocess)


w2v_model = api.load("word2vec-google-news-300") 
print("Word2Vec model loaded.")


def get_vector(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

df['vector'] = df['tokens'].apply(lambda x: get_vector(x, w2v_model))

X = np.vstack(df['vector'].values)
y = df['airline_sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")


def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess(tweet)
    vec = get_vector(tokens, w2v_model).reshape(1, -1)
    return model.predict(vec)[0]

# Example
tweet_example = "I am so frustrated with the flight delays!"
print("Example prediction:", predict_tweet_sentiment(clf, w2v_model, tweet_example))
