# coding: utf-8

#usr/bin/python3

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import TweetTokenizer
import pandas as ps
import re

def add_sentiment_analysis(corpus, data):
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
        m4.append(abs(compound))

    print("Finished adding features")
    print("Discretizing...")

    m1 = ps.cut(m1,6,labels=False)
    m2 = ps.cut(m2,6,labels=False)
    #m3 = ps.cut(m3,6,labels=False)
    m4 = ps.cut(m4,6,labels=False)

    print("Finished discretizing features")
    
    for i in range(len(data)):
        data[i].append(m1[i]/10)
        data[i].append(m2[i]/10)
        #data[i].append(m3[i]/10)
        data[i].append(m4[i]/10)

    return data        

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

def index_corpus_words(data):
    tok = TweetTokenizer()
    #lem = WordNetLemmatizer()
    frequency = {}
    vocabulary = {}
    word_index = 2
    for row in data['Tweet text'].values:
        for word in tok.tokenize(row.lower()):
            #word = lem.lemmatize(word)
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

    positive_emoji = list(filter(None,open(
        'positive_emoji', 'r').read().split('\n')))
    negative_emoji = list(filter(None,open(
        'negative_emoji', 'r').read().split('\n')))
    #Replace URLs by <website>:
    line = re.sub(r"https?://.+"," <website> ",line)
    #Replace usernames by <user>:
    line = re.sub(r"@[^\s]*"," <user> ",line)
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
    line = re.sub(r"([#+|.+|=+|,+|\/+|\?+|\!+|\(+|\)+|\[+|\]+|\"+|\|+])",
        r" \1 ", line)
    #Replace repeating characters
    line = re.sub(r'(.)\1+', r'\1\1', line)
    #Delete remaining punctuation
    line = re.sub(r'(,|\\|\/|\*|\'|\"|\~)', "", line)
    #Lemmatize words
    new_line = []
    lem = WordNetLemmatizer()
    new_line = " ".join(lem.lemmatize(word) for word in line.split())
    return new_line

