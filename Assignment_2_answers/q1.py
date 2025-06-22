import pandas as pd
import numpy as np
import nltk
import gensim.downloader as api
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import warnings

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
df.columns = ["Label", "Message"]

stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

df['Tokens'] = df['Message'].apply(preprocess)

w2v_model = api.load('word2vec-google-news-300')  
print("Model loaded.")


def get_vector(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

df['Vector'] = df['Tokens'].apply(lambda x: get_vector(x, w2v_model))

X = np.vstack(df['Vector'].values)
y = df['Label'].apply(lambda x: 1 if x == 'spam' else 0)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")


def predict_message_class(model, w2v_model, message):
    tokens = preprocess(message)
    vec = get_vector(tokens, w2v_model).reshape(1, -1)
    prediction = model.predict(vec)[0]
    return "spam" if prediction == 1 else "ham"


example_msg = "Congratulations! You've won a free iPhone. Click now to claim."
print(f"Predicted class for example: {predict_message_class(clf, w2v_model, example_msg)}")
