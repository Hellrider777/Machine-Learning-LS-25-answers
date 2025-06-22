import math
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Define the corpus
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]



tokenized = [doc.lower().split() for doc in corpus]
vocab = sorted(set(word for doc in tokenized for word in doc))


tf = []
for doc in tokenized:
    doc_tf = defaultdict(float)
    for word in doc:
        doc_tf[word] += 1
    for word in doc_tf:
        doc_tf[word] /= len(doc)  
    tf.append(doc_tf)


df = defaultdict(int)
for word in vocab:
    df[word] = sum(1 for doc in tokenized if word in doc)


N = len(corpus)
idf = {word: math.log(N / df[word]) for word in vocab}


tfidf_manual = []
for doc_tf in tf:
    doc_vec = []
    for word in vocab:
        tfidf_score = doc_tf[word] * idf[word] if word in doc_tf else 0.0
        doc_vec.append(round(tfidf_score, 4))
    tfidf_manual.append(doc_vec)


count_vec = CountVectorizer()
X_count = count_vec.fit_transform(corpus).toarray()
count_vocab = count_vec.get_feature_names_out()


tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(corpus).toarray()
tfidf_vocab = tfidf_vec.get_feature_names_out()



print("\nVocabulary")
print("Manual Vocab:       ", vocab)
print("CountVectorizer Vocab:", list(count_vocab))
print("TfidfVectorizer Vocab:", list(tfidf_vocab))

print("\nManual TF-IDF")
for i, vec in enumerate(tfidf_manual):
    print(f"Doc {i+1}: {dict(zip(vocab, vec))}")

print("\nCountVectorizer (Sklearn)")
for i, row in enumerate(X_count):
    print(f"Doc {i+1}: {dict(zip(count_vocab, row))}")

print("\nTfidfVectorizer (Sklearn)")
for i, row in enumerate(X_tfidf):
    print(f"Doc {i+1}: {dict(zip(tfidf_vocab, np.round(row, 4)))}")

