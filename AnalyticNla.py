import numpy as np
import pandas as pd
import re
import nltk
import string

# nltk.download()

# https://www.nltk.org/book/ch07.html#ref-ie-segment
# https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/

# Lire les données depuis le fichier CSV
rawData = pd.read_csv("timetables.csv", sep=';', names=['idTrajet', 'Trajet', 'TempsTrajet'], header=1)


# Function pour retirer la ponctuation
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])  # retire toute les ponctuations
    return text_nopunct


rawData['body_text_clean'] = rawData['Trajet'].apply(lambda x: remove_punct(x))

def tokenize(text):
    tokens = re.split('\W+', text) #W+ veux dire sois un caratère A-Za-z0-9 ou un tiret
    return tokens


rawData['body_text_tokenized'] = rawData['body_text_clean'].apply(lambda x: tokenize(x.lower()))

stopword = nltk.corpus.stopwords.words('french')

#Function pour retirer les stopwords
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]#suppression des stopwords
    return text


rawData['body_text_nostop'] = rawData['body_text_tokenized'].apply(lambda x: remove_stopwords(x))

