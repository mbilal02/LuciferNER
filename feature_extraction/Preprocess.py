import csv
import string
from itertools import groupby

import re

from collections import defaultdict as ddict
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer


##########################################Calculate vocab size##########################################################################
def vocab_size(tweets):
    t = Tokenizer()
    t.fit_on_texts(tweets)
    size = len(t.word_index) + 1
    return size


################################# Ecoding the documents######################################################################
def encode_doc(sample):
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(sample)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(sample)
    print(encoded_docs)

    # pad documents to a max length of 4 words
    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs


##############################################################
# Input and output functions
##############################################################

def read_file_as_list_of_tuples(filename, delimiter='\t'):
    """It returns a list of tweets, and each tweet is a tuple of the elements found in the line"""
    with open(filename) as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        return [list(tuple(e) for e in g) for k, g in groupby(reader, lambda x: not x) if not k]


def read_file_as_lists(filename, delimiter='\t'):
    with open(filename, encoding='utf8', errors='ignore') as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        labeled_tokens = [zip(*g) for k, g in groupby(reader, lambda x: not [s for s in x if s.strip()]) if not k]
        tokens, labels = zip(*labeled_tokens)
        return [list(t) for t in tokens], [list(l) for l in labels]


def read_datasets(filename):
    tweets_train, labels_train = read_file_as_lists(filename)
    tweets_dev, labels_dev = read_file_as_lists(filename)
    tweets_test, labels_test = read_file_as_lists(filename)

    # Combining train and dev to account for different domains
    tweets_train += tweets_dev
    labels_train += labels_dev

    return (tweets_train, labels_train), (tweets_test, labels_test)


def read_test_datasets(file):
    tweet_test, label = read_file_as_lists(file)

    return tweet_test, label


def encode_tokens(token2index, tokens):
    return [[token2index[tkn] for tkn in tkns] for tkns in tokens]


def flatten(l):
    """Flatten 2D lists"""
    return [i for sublist in l for i in sublist]


######## Orthographic encoding #################################

def orthigraphic_char(ch):
    try:
        if re.match('[a-z]', ch):
            return 'c'
        if re.match('[A-Z]', ch):
            return 'C'
        if re.match('[0-9]', ch):
            return 'n'
        if ch in string.punctuation:
            return 'p'
    except TypeError:
        print('TypeError:', ch)
    return 'x'


def orthographic_tweet(tweet):
    return [''.join([orthigraphic_char(ch) for ch in token]) for token in tweet]


def orthographic_mapping(tweets):
    return [orthographic_tweet(tweet) for tweet in tweets]


def match_up_to(x, elems):
    acc = []
    for e in elems:
        acc.append(e)
        if x == ''.join(acc):
            return len(acc)
    return None


def encode_orthography(tweets, max_len):
    index2ortho = ['x', 'c', 'C', 'n', 'p']
    ortho2index = ddict(lambda: 0, {o: i for i, o in enumerate(index2ortho)})
    encoded_ortho = orthographic_mapping(tweets)
    encoded_ortho = encode_tokens(ortho2index, flatten(encoded_ortho))
    encoded_ortho = pad_sequences(encoded_ortho, maxlen=max_len)
    return encoded_ortho
