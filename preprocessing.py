# coding: utf-8

#usr/bin/python3

from nltk.stem.wordnet import WordNetLemmatizer
from keras.preprocessing import sequence
import re


def index_corpus_words(data_train):
    frequency = {}
    vocabulary = {}
    word_index = 1
    for row in data_train['Tweet text'].values:
         for word in normalize(row).split():
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

def map_words_and_add_padding(vocabulary, corpus, maxlen):
    new_corpus = []
    for row in corpus:
        new_row = []
        for word in row:
            if word in vocabulary.keys():
                new_row.append(vocabulary[word])
            else:
                new_row.append(vocabulary["<unknown>"])
        new_corpus.append(new_row)
    return sequence.pad_sequences(new_corpus, maxlen=maxlen)

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
