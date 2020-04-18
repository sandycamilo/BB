import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)

# print(data["intents"])

words = []
labels = []
docs = []

for intent in data["intents"]: # loops through every intent(dictionary) in intents 
    for pattern in intent["patterns"]: # access dictionary in intent and gets 'patterns'
        wrds = nltk.word_tokenize(pattern) # gets every word in patterns- returns a list with all different words in it
        words.extend(wrds) # places all tokenized words into words list by extending the list 'wrds' to words
        docs.append(pattern) # adds pattern to docs []

    # get all tags in intent
    if intent["tag"] not in labels: 
        labels.append(intent["tag"])