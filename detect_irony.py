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

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import dump_svmlight_file
from sklearn import metrics


#parameters
max_features = 5000
maxlen = 100
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10

seed = 7 # fix random seed for reproducibility
number_of_splits = 5

def load_data(filename):
    data = ps.read_csv(filename, sep="\t")
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
    kfold = StratifiedKFold(n_splits=number_of_splits, shuffle=True, random_state=seed)
    cvscores = []
    split_num = 0
    for train, test in kfold.split(data_corpus, data_labels):
        split_num += 1
        print("Split no.: " + str(split_num))
        # create model
        model = build_model(data_corpus[train], data_labels[train])
        # evaluate the model
        scores = model.evaluate(data_corpus[test], data_labels[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

if __name__ == "__main__":
    data = load_data('./SemEval2018-T3-train-taskA.txt')
    vocabulary = pre.index_corpus_words(data)
    data_labels = np.array(data['Label'].values)
    
    # Normalize - needs check
    #data_corpus = pre.map_words(vocabulary, pre.normalize_corpus(['Tweet text'].values))
    #data_corpus = pre.add_sentiment_analysis(pre.normalize_corpus(data['Tweet text'].values), data_corpus)

    data_corpus = pre.map_words(vocabulary, data['Tweet text'].values)
    data_corpus = pre.add_sentiment_analysis(data['Tweet text'].values, data_corpus)
    
    # Padding
    data_corpus = sequence.pad_sequences(data_corpus, maxlen=maxlen)
   
    crossvalidation(data_corpus, data_labels)


    #K_FOLDS = 10 # 10-fold crossvalidation
    #CLF = LinearSVC() # the default, non-parameter optimized linear-kernel SVM

    ## Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
    #predicted = cross_val_predict(CLF, data_corpus, data_labels, cv=K_FOLDS)
    
    #score = metrics.accuracy_score(data_labels, predicted)
    #print ("acc:", score)
    ##for p in predicted:
    ##    PREDICTIONSFILE.write("{}\n".format(p))
    ##PREDICTIONSFILE.close()