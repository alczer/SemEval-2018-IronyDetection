#usr/bin/python3
#
#
#
#install:
#keras,nltk

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

#parameters
max_features = 5000
maxlen = 100
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10

def add_sentiment_analysis_and_add_padding(corpus, data):
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    print("Adding features...")
    for i in range(len(corpus)):
        row = []
        level, neg, pos, neu, compound = sentiment_analyse(corpus[i])
        lvl1, lvl2 = sentiment_analyse_2(corpus[i])
        m1.append(lvl1)
        m2.append(lvl2)
        m3.append(level)
        m4.append(compound)

    print("Finished adding features")
    print("Discretizing...")

    m1 = ps.cut(m1,6,labels=False)
    m2 = ps.cut(m2,6,labels=False)
    m3 = ps.cut(m3,6,labels=False)
    m4 = ps.cut(m4,6,labels=False)

    print("Finished discretizing features")
    
    for i in range(len(data)):
        data[i].append(m1[i]/10)
        data[i].append(m2[i]/10)
        data[i].append(m3[i]/10)
        data[i].append(m4[i]/10)

    return sequence.pad_sequences(data, maxlen=maxlen)    

    

def sentiment_analyse(tweet):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(tweet)
    neg = sentiment_scores['neg']
    pos = sentiment_scores['pos']
    neu = sentiment_scores['neu']
    compound = sentiment_scores['compound']
    level = neg*pos    
    #print(sentiment_scores)
    #print(level)
    return level, neg, pos, neu, compound


def sentiment_analyse_2(tweet):
    sid = SentimentIntensityAnalyzer()

    split = tweet.replace('_',' ').replace('#',' ').replace(':',' ').split(" ")
    half1 = ""
    half2 = ""
    iterator = 0
    for word in split:
        if iterator < len(split)/2:
            half1 += word + " "
        else:
            half2 += word + " "
        iterator += 1

    sentiment_scores_1 = sid.polarity_scores(half1)
    sentiment_scores_2 = sid.polarity_scores(half2)

    neg1 = sentiment_scores_1['neg']
    pos1 = sentiment_scores_1['pos']
    neu1 = sentiment_scores_1['neu']
    compound1 = sentiment_scores_1['compound']

    neg2 = sentiment_scores_1['neg']
    pos2 = sentiment_scores_1['pos']
    neu2 = sentiment_scores_1['neu']
    compound2 = sentiment_scores_1['compound']
    
    lvl1 = (pos1+neg1)*(pos2+neg2)
    lvl2 = max(pos1*neg2,pos2*neg1)

    return lvl1, lvl2


def load_and_split_data(filename):
    data = ps.read_csv(filename, sep="\t")
    return np.split(data, [int(.8*len(data))])

def index_corpus_words(data_train):
    tok = TweetTokenizer()
    frequency = {}
    vocabulary = {}
    word_index = 2
    for row in data_train['Tweet text'].values:
        for word in tok.tokenize(row.lower()):
            if word not in frequency.keys():
                frequency[word]=1
            else:
                frequency[word]+=1

    for word in frequency.keys():
        if frequency[word] > 1:
            vocabulary[word] = word_index
            word_index += 1
    vocabulary["<unknown>"] = word_index
    return vocabulary

def map_words(vocabulary, corpus):
    new_corpus = []
    for row in corpus:
        new_row = []
        for word in row:
            if word in vocabulary.keys():
                new_row.append(vocabulary[word])
            else:
                new_row.append(vocabulary["<unknown>"])

        new_corpus.append(new_row)
    return new_corpus 

def build_model(data_train_corpus,data_train_labels):
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

def evaluate(model,data_test_corpus,data_test_labels):
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
    data_train, data_test = load_and_split_data('./SemEval2018-T3-train-taskA.txt')
    vocabulary = index_corpus_words(data_train)
    data_train_labels = np.array(data_train['Label'].values)
    data_test_labels = np.array(data_test['Label'].values)
    
    data_train_corpus = map_words(vocabulary, data_train['Tweet text'].values)
    data_train_corpus = add_sentiment_analysis_and_add_padding(data_train['Tweet text'].values, data_train_corpus)

    data_test_corpus = map_words(vocabulary, data_test['Tweet text'].values)
    data_test_corpus = add_sentiment_analysis_and_add_padding(data_test['Tweet text'].values, data_test_corpus)

    model = build_model(data_train_corpus, data_train_labels)
    evaluate(model,data_test_corpus,data_test_labels)