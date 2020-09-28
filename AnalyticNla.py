import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
#Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import MultinomialNB

#https://www.nltk.org/book/ch07.html#ref-ie-segment
#https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/

nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_csv('timetables.csv', sep=';')

# tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
text_counts = cv.fit_transform(data['trajet'])

X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['trajet'], test_size=0.3, random_state=1)

# Model Generation Using Multinomial Naive Bayes

clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))