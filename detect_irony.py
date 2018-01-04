# coding: utf-8

#usr/bin/python3
#
#
#
#install:
#keras,nltk, nltk.download('wordnet')

import preprocessing as pre
import pandas as ps
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


#parameters

max_features = 10000
maxlen = 200
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10

def load_and_split_data(filename):
    data = ps.read_csv(filename, sep="\t")
    return np.split(data, [int(.8*len(data))])

def build_model(data_train_corpus, data_train_labels):
    model = Sequential()
    model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(data_train_corpus, data_train_labels,
        batch_size=batch_size,
        epochs=epochs)
    return model

def evaluate(model, data_test_corpus, data_test_labels):
    results = model.predict(data_test_corpus)
    results_binary = []
    for result in results:
        if result < 0.5:
            results_binary.append(0)
        else:
            results_binary.append(1)
    nbr_test_passed = 0
    nbr_tests = 0
    for result, label in zip(results_binary, data_test_labels):
        if result == label:
            nbr_test_passed+=1
        nbr_tests+=1
    print("Number of tests passed:")
    print(nbr_test_passed)
    print("Number of tests:")
    print(nbr_tests)
    print("\n")
    print("Accuracy:")
    print(nbr_test_passed/nbr_tests)

if __name__ == "__main__":
    data_train, data_test = load_and_split_data('data')
    vocabulary = pre.index_corpus_words(data_train)
    data_train_labels = np.array(data_train['Label'].values)
    data_test_labels = np.array(data_test['Label'].values)
    data_train_corpus = pre.map_words_and_add_padding(vocabulary, 
        data_train['Tweet text'].values,maxlen)
    data_test_corpus = pre.map_words_and_add_padding(vocabulary, 
        data_test['Tweet text'].values, maxlen)
    model = build_model(data_train_corpus, data_train_labels)
    evaluate(model, data_test_corpus, data_test_labels)
