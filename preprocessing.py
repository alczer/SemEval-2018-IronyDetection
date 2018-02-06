# coding: utf-8

#usr/bin/python3

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import TweetTokenizer
import pandas as ps
import re

def add_sentiment_analysis(corpus, data, mode):
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    print("Adding features...")
    if mode == 1 or mode == 2:
        for i in range(len(corpus)):
            lvl1, lvl2 = sentiment_analyse(corpus[i])
            m1.append(lvl1)
            m2.append(lvl2)
        print("Finished adding features")
        print("Discretizing...")
        #m1 = ps.qcut(m1,6, labels=False, duplicates = 'drop')
        #m2 = ps.qcut(m2,6, labels=False, duplicates = 'drop')
        m1 = ps.cut(m1,8, labels=False)
        m2 = ps.cut(m2,8, labels=False)
        
    elif mode == 3 or mode == 4:
        for i in range(len(corpus)):
            level, neg, pos, neu, compound = sentiment_analyse_simple(corpus[i])
            m3.append(level)
            m4.append(compound)
        print("Finished adding features")
        print("Discretizing...")
        m1 = ps.cut(m3,8, labels=False)
        m2 = ps.cut(m4,8, labels=False)
        
    print("Finished discretizing features")

    if mode == 1:
        for i in range(len(data)):
            data[i].append(m1[i]/10)
    elif mode == 2:
        for i in range(len(data)):
            data[i].append(m2[i]/10)
    elif mode == 3:
        for i in range(len(data)):
            data[i].append(m3[i]/10)
    elif mode == 4:
        for i in range(len(data)):
            data[i].append(m4[i]/10)

    
    measures = {}
    if mode == 1 or mode == 2:
        measures = {"M1":m1,"M2":m2}
    elif mode == 3 or mode == 4:
        measures = {"M1":m3,"M2":m4}
    
    return data, measures        

def sentiment_analyse_simple(tweet):
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

def sentiment_analyse(tweet):
    sid = SentimentIntensityAnalyzer()

    tweet = normalize(tweet)
    split = tweet.split(" ")
    half1 = ""
    half2 = ""
    iterator = 0
    for word in split:
        if iterator < len(split)/2:
            half1 += word + " "
        else:
            half2 += word + " "
        iterator += 1

    #print(half1)
    #print(half2)
    sentiment_scores_1 = sid.polarity_scores(half1)
    sentiment_scores_2 = sid.polarity_scores(half2)    

    neg1 = sentiment_scores_1['neg']
    pos1 = sentiment_scores_1['pos']
    #neu1 = sentiment_scores_1['neu']
    #compound1 = sentiment_scores_1['compound']

    neg2 = sentiment_scores_1['neg']
    pos2 = sentiment_scores_1['pos']
    #neu2 = sentiment_scores_1['neu']
    #compound2 = sentiment_scores_1['compound']


    if "<positive_emoji>" in half1: 
        #print("found positive")
        if pos1 < 0.5:
            pos1 += 0.5
        else:
            pos1 = 1.0
    if "<negative_emoji>" in half1: 
        #print("found negative")
        if neg1 < 0.5:
            neg1 += 0.5
        else:
            neg1 = 1.0
    if "<dots>"  in half1: 
        #print("found negative")
        if neg1 < 0.5:
            neg1 += 0.5
        else:
            neg1 = 1.0
    if "<positive_emoji>" in half2: 
        #print("found positive")
        if pos2 < 0.5:
            pos2 += 0.5
        else:
            pos2 = 1.0
    if "<negative_emoji>" in half2: 
        #print("found negative")
        if neg2 < 0.5:
            neg2 += 0.5
        else:
            neg2 = 1.0
    if "<dots>"  in half2: 
        #print("found negative")
        if neg2 < 0.75:
            neg2 += 0.25
        else:
            neg2 = 1.0

    lvl1 = (pos1+neg1)*(pos2+neg2)
    lvl2 = max(pos1+neg2/2, pos2+neg1/2)

    #if "\"" in tweet: 
    #    #print("found negative")
    #    if lvl1 < 0.6:
    #        lvl1 += 0.4
    #    else:
    #        lvl1 = 1.0
                
    #    if lvl2 < 0.6:
    #        lvl2 += 0.4
    #    else:
    #        lvl2 = 1.0

    #print("LVL1:" + str(lvl1) + " LVL2:" + str(lvl2))
    #print()
    return lvl1, lvl2

def index_corpus_words(data):
    tok = TweetTokenizer() # improves results
    lem = WordNetLemmatizer() # slightly / no change
    frequency = {}
    vocabulary = {}
    word_index = 2
    for row in data['Tweet text'].values:
        #row = normalize(row)
        for word in tok.tokenize(row.lower()): # TODO: move
            word = lem.lemmatize(word)  # TODO: move
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


def normalize_corpus(corpus):
    new_corpus = []
    for line in corpus:
        new_corpus.append(normalize(line))
    return(corpus)

def normalize(line):
    #print(line)

    #Replace URLs by <website>:
    line = re.sub(r"https?://.+"," <website> ",line)
    #Replace usernames by <user>:
    line = re.sub(r"@[^\s]*"," <user> ",line)
    
    #EMOJIS:
    positive_emoji = list(filter(None,open('positive_emoji', 'r').read().split('\n')))
    negative_emoji = list(filter(None,open('negative_emoji', 'r').read().split('\n')))

    #Separate emojis and change the text to lowercase
    line = re.sub(r"(:[a-z_\-]+:)",r" \1 ", line).lower()

    #Replace positive emojis by <positive_emoji>
    line = " ".join([" <positive_emoji> " if word in positive_emoji
        else word for word in line.split()])
    line = re.sub(r"\s((:|;)?[\-|\~]\*|<3(_)+<3|<3)(\s|$)",
        " <positive_emoji> ",line)
    line = re.sub(r"(:|;)[\-|\~]?(p|d|P)(\s|$)*"," <positive_emoji> ",
        line)
    line = re.sub(r"\s(((:|;|\||x)[']?[\-|\~]?(\)+|D+|\}+|\]+|>+)|\^_?\^|"
                  "haha[ha]*|(\[+|\{+|\(+)[\-|\~]?(;|;|\|)))(\s|$)",
        " <positive_emoji> ",line)

    #Replace negative emojis by <negative_emoji>
    line = " ".join([" <negative_emoji> " if word in negative_emoji
        else word for word in line.split()])
    line = re.sub(r"((:|;)[']?[\-|\~]?(\(+|C+|\[+|\|+|\{+|s|S|<+)|"
        "T(_)+T|;(_)+;|\-(_)+\-|(\)|\}|\])[\-|\~]?(;|;))(\s|$)",
        " <negative_emoji> ",line)

    #Replace remaining emojis by <neutral_emoji>
    line = re.sub(r"(<(<)+|(<)+(\-)+|>(>)+|(\-)+(>)+)",
        " <neutral_emoji> ",line)
    line = re.sub(r"((:|;)[\-|\~]?(o|O)|O\.o|o\.O|\*\-\*)(\s|$)",
        " <neutral_emoji> ",line)
    line = re.sub(r":[a-z_\-]+:"," <neutral_emoji> ",line)

    #Separate punctuation
    line = re.sub(r"([#+|.+|=+|,+|\/+|\?+|\!+|\(+|\)+|\[+|\]+|\"+|\|+])", r" \1 ", line)

    #Remove numbers
    line = re.sub(r"[0-9]*", "",line)
    #Replace repeating characters
    #line = re.sub(r'(.)\1+', r'\1\1', line)
    #Delete remaining punctuation
    #line = re.sub(r'(,|\\|\/|\*|\'|\"|\~)', "", line)

    #replace hastags by space, remove unnecesarry spaces
    line = line.replace('#',' ')
    line = re.sub(r' +', r' ', line)

    #dots
    line = line.replace(". . .","<dots>")
    line = line.replace(". .","<dots>")


    new_line = []
    #Lemmatize words
    #lem = WordNetLemmatizer()
    #new_line = " ".join(lem.lemmatize(word) for word in line.split())
    new_line = " ".join(word for word in line.split())

    #print(new_line)
    return new_line

