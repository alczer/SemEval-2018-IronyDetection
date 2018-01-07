#usr/bin/python3
#
#
#
#install:
#keras,nltk

import preprocessing as pre
import pandas as ps
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import codecs

from sklearn.model_selection import StratifiedKFold


#parameters
max_features = 5000
maxlen = 100
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10

seed = 7 # useless when shuffle = False
number_of_splits = 5

def load_data(filename):
    #data = ps.read_csv(filename, sep="\t") # faulty - misses several tweets
    doc = codecs.open(filename,'rU','UTF-8') # open for reading with "universal" type set
    dict = {"Tweet index": [], "Label": [], "Tweet text": []}
    lines = []
    lines = doc.readlines()
    del lines[0] # delete header line

    for x in lines:
        y=x.split('\t')
        dict["Tweet index"].append(y[0])
        dict["Label"].append(y[1])
        dict["Tweet text"].append(y[2])
    
    data = ps.DataFrame(dict)
    return data

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

def crossvalidation(data_corpus, data_labels):
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=number_of_splits)#, shuffle=True, random_state=seed)
    cvscores = []
    cvresults = []
    split_num = 0
    for train, test in kfold.split(data_corpus, data_labels):
        split_num += 1
        print("Split no. " + str(split_num) + "/" + str(number_of_splits))
        # create model
        model = build_model(data_corpus[train], data_labels[train])
        # evaluate the model
        scores = model.evaluate(data_corpus[test], data_labels[test], verbose=0)
        results = model.predict(data_corpus[test])
        cvresults.extend(results)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    results_binary = []
    for result in cvresults:
        if result < 0.5:
            results_binary.append(0)
        else:
            results_binary.append(1)
    nbr_test_passed = 0
    nbr_tests = 0
    for result, label in zip(results_binary, data_labels):
        if str(result) == str(label):
            nbr_test_passed+=1
        nbr_tests+=1
    print("Number of tests passed:")
    print(nbr_test_passed)
    print("Number of tests:")
    print(nbr_tests)
    
    print("Accuracy:")
    print(nbr_test_passed/nbr_tests)

    PREDICTIONSFILE = open("results1.txt", "w")
    for p in cvresults:
        if p < 0.5:
            PREDICTIONSFILE.write("{}\n".format(0))
        else:
            PREDICTIONSFILE.write("{}\n".format(1))
    PREDICTIONSFILE.close()

if __name__ == "__main__":
    data = load_data('./SemEval2018-T3-train-taskA.txt')
    vocabulary = pre.index_corpus_words(data)
    data_labels = np.array(data['Label'].values)
    print(len(data_labels))

    # Normalize - needs check
    #data_corpus = pre.map_words(vocabulary, pre.normalize_corpus(['Tweet text'].values))
    #data_corpus = pre.add_sentiment_analysis(pre.normalize_corpus(data['Tweet text'].values), data_corpus)

    data_corpus = pre.map_words(vocabulary, data['Tweet text'].values)
    data_corpus = pre.add_sentiment_analysis(data['Tweet text'].values, data_corpus)
    
    # Padding
    data_corpus = sequence.pad_sequences(data_corpus, maxlen=maxlen)
   
    crossvalidation(data_corpus, data_labels)

