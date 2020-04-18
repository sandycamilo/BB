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
docs_x = []
docs_y = [] # each pattern we have correlates to a specific tag

for intent in data["intents"]: # loop through every intent(dictionary) in intents 
    for pattern in intent["patterns"]: # access dictionary in intent and gets 'patterns'
        pattern_words = nltk.word_tokenize(pattern) # get every word in patterns- returns a list with all different words in it
        words.extend(pattern_words) # place all tokenized words into words list by extending the list 'wrds' to words
        docs_x.append(pattern) # add pattern to docs_x []
        docs_y.append(intent["tag"]) # helps classify each pattern

    # get all tags in intent
    if intent["tag"] not in labels: 
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words] # convert all words into lowercase 
words = sorted(list(set(words))) # remove duplicate elements and sorts the words - all in a list

labels = sorted(labels) # sort labels 

# how many times does each word occur - 
# frequency goes with entry ~ matching entry with count

training = [] 
output = []
out_empty = [0 for _ in range(len(labels))]  

for x, doc in enumerate(docs_x):
    bag = []  

    pattern_words = [stemmer.stem(w) for w in doc] # stem all words that are in patterns  

    for w in words:
        if w in pattern_words: # if words exists in the current pattern that we are looping through
            bag.append(1) # the word exists so we place a 1 , no matter how many times it exists
        else:
            bag.append(0) # the word is not there so we place a 0

    output_row = out_empty[:] 
    output_row[labels.index(docs_y[x])] = 1  # look through labels list & check where the tag is in that list then set that value to 1 in output row
                                            
    training.append(bag) 
    output.append(output_row)

training = numpy.array(training) 
output = np.array(output)