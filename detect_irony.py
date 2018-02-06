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
epochs = 7

#seed = 7 # useless when shuffle = False
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

def load_test_data(filename):
    #data = ps.read_csv(filename, sep="\t") # faulty - misses several tweets
    doc = codecs.open(filename,'rU','UTF-8') # open for reading with "universal" type set
    dict = {"Tweet index": [], "Tweet text": []}
    lines = []
    lines = doc.readlines()
    del lines[0] # delete header line

    for x in lines:
        y=x.split('\t')
        dict["Tweet index"].append(y[0])
        dict["Tweet text"].append(y[1])
    
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
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(data_train_corpus, data_train_labels,
        batch_size=batch_size,
        epochs=epochs)
    return model

def predict(data_corpus, data_labels, data_test_corpus):
    model = build_model(data_corpus, data_labels)
    results = model.predict(data_test_corpus)
    
    PREDICTIONSFILE = open("predictions-taskA.txt", "w")
    for p in results:
        if p < 0.5:
            PREDICTIONSFILE.write("0\n")
        else:
            PREDICTIONSFILE.write("1\n")
    PREDICTIONSFILE.close()
    return results

def crossvalidation(data_corpus, data_labels):
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

    PREDICTIONSFILE = open("results.txt", "w")
    for p in cvresults:
        if p < 0.5:
            PREDICTIONSFILE.write("0\n")
        else:
            PREDICTIONSFILE.write("1\n")
    PREDICTIONSFILE.close()

    return cvresults

def write_results_file_2(results, test_data, test_measures):
    INFOFILE = open("results2.txt", "w", encoding = 'utf-8')
    #print(len(zip(results, test_data['Tweet text'].values, test_measures["M2"])))
    for result, text, M2 in zip(results, test_data['Tweet text'].values, test_measures["M2"]):
        if result < 0.5:
            output = "0, prob:" + str(result) + " M2:" + str(M2) + " " + pre.normalize(text) + "\n"
            INFOFILE.write(output)
        else:
            output = "1, prob:" + str(result) + " M2:" + str(M2) + " " + pre.normalize(text) + "\n"
            INFOFILE.write(output)
    INFOFILE.close()

def write_results_file(results, data, measures):
    PREDICTIONSFILE = open("results1.txt", "w", encoding = 'utf-8')
    for result, label, text, M1, M2 in zip(results, data['Label'].values, data['Tweet text'].values, measures["M1"], measures["M2"]):
        if result < 0.5:
            output = "label:" + str(label) + "/0 prob:" + str(result) + " M1:" + str(M1) + " M2:" + str(M2) + " " + pre.normalize(text) + "\n"
            PREDICTIONSFILE.write(output)
        else:
            output = "label:" + str(label) + "/1 prob:" + str(result) + " M1:" + str(M1) + " M2:" + str(M2) + " " + pre.normalize(text) + "\n"
            PREDICTIONSFILE.write(output)
    PREDICTIONSFILE.close()

if __name__ == "__main__":
    # Load data
    test_data = load_test_data('./SemEval2018-T3_input_test_taskA.txt')
    data = load_data('./SemEval2018-T3-train-taskA.txt')

    # Index words
    vocabulary = pre.index_corpus_words(data)
    data_labels = np.array(data['Label'].values)

    print(len(data_labels))

    # Normalize and map words
    data_corpus = pre.map_words(vocabulary, pre.normalize_corpus(data['Tweet text'].values))
    data_test_corpus = pre.map_words(vocabulary, pre.normalize_corpus(test_data['Tweet text'].values))

    # Add sentiment analysis
    data_corpus, measures = pre.add_sentiment_analysis(data['Tweet text'].values, data_corpus, 2)
    data_test_corpus, test_measures = pre.add_sentiment_analysis(test_data['Tweet text'].values, data_test_corpus, 2)

    # Padding
    data_corpus = sequence.pad_sequences(data_corpus, maxlen=maxlen)
    data_test_corpus = sequence.pad_sequences(data_test_corpus, maxlen=maxlen)

    # Test results
    results = predict(data_corpus, data_labels, data_test_corpus)
    write_results_file_2(results, test_data, test_measures)

    # Crossvalidation results
    results2 = crossvalidation(data_corpus, data_labels)
    write_results_file(results2, data, measures)

