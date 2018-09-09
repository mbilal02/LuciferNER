import csv
import string
from itertools import groupby

from nltk.corpus.reader.conll import ConllCorpusReader

import re



######################################################
#                     Load Data                      #
######################################################

def conllReader(corpus):
    root = "../data/"

    ccorpus = ConllCorpusReader(root, ".conll", ('words', 'pos', 'tree'))

    return ccorpus.tagged_sents(corpus)

def load_sentence(train_name, dev_name, test_name):
    """
    reads in a way that every line contain a word and it's tag
    every sentence is split with a empty line.

    """

    img_id = []
    sentences = []
    sentence = []
    sent_maxlen = 0
    word_maxlen = 0
    datasplit = []

    for fname in (train_name, dev_name, test_name):
        datasplit.append(len(img_id))
        with open(fname, 'r', encoding='utf8') as file:
            for line in file:
                line = line.rstrip()
                if line == '':
                    sent_maxlen = max(sent_maxlen, len(sentence))
                    sentences.append(sentence)
                    sentence = []
                else:
                    sentence.append(line.split('\t'))
                    word_maxlen = max(word_maxlen, len(str(line.split()[0])))

    sentences.append(sentence)
    num_sentence = len(sentences)

    return [datasplit, sentences, sent_maxlen, word_maxlen, num_sentence]

def read_file_as_list_of_tuples(filename, delimiter='\t'):
    """It returns a list of tweets, and each tweet is a tuple of the elements found in the line"""
    with open(filename, encoding='utf8') as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        return [list(tuple(e) for e in g) for k, g in groupby(reader, lambda x: not x) if not k]


def read_file_as_lists(filename, delimiter='\t'):
    with open(filename, encoding='utf8') as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        labeled_tokens = [zip(*g) for k, g in groupby(reader, lambda x: not [s for s in x if s.strip()]) if not k]
        tokens, labels = zip(*labeled_tokens)
        return [list(t) for t in tokens], [list(l) for l in labels]

def unzip(list_of_tuples):
    return [list(elem) for elem in zip(*list_of_tuples)]


def flatten_rec(l):
    # TODO: fix problem
    if not l:
        return []
    if isinstance(l[0], list):
        return flatten(l[0]) + flatten(l[1:])
    return l[:1] + flatten(l[1:])


def flatten(l):
    """Flatten 2D lists"""
    return [i for sublist in l for i in sublist]



######################################################
#   Character Encoding                               #
######################################################

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










