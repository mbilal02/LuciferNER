import nltk as nltk

import tensorflow as tf
import numpy as np


class Data(enumerate):
    TRAINING = "training"
    TEST = "test"


class WordFeatures(object):
    def __init__(self, word_features="", char_features=""):

        self.word_feature = word_features
        self.char_features = char_features

    @staticmethod
    def load_data(data):
        if data == Data.TRAINING:
            sents = open("../data/train.txt").read()
            return sents

        if data == Data.TEST:
            t_sent = open("../data/train.txt").read()
            return t_sent
        else:
            print("File not found")

    def word_representation(self):

        PAD_TOKEN = 0

        word2idx = {
            'PAD': PAD_TOKEN}  # dict so we can lookup indices for tokenising our text later from string to sequence of integers
        weights = []

        with open("../data/glove.twitter.27B.100d.txt", 'r') as file:
            for index, line in enumerate(file):
                values = line.split()  # Word and weights separated by space
                word = values[0]  # Word is first symbol on each line
                word_weights = np.asarray(values[1:], dtype=np.float32)  # Remainder of line is weights for word
                word2idx[word] = index + 1  # PAD is our zeroth index so shift by one
                weights.append(word_weights)

                if index + 1 == 40000:
                    # Limit vocabulary to top 40k terms
                    break

                EMBEDDING_DIMENSION = len(weights[0])
                # Insert the PAD weights at index 0 now we know the embedding dimension
                weights.insert(0, np.random.randn(EMBEDDING_DIMENSION))

                # Append unknown and pad to end of vocab and initialize as random
                UNKNOWN_TOKEN = len(weights)
                word2idx['UNK'] = UNKNOWN_TOKEN
                weights.append(np.random.randn(EMBEDDING_DIMENSION))

                # Construct our final vocab
                weights = np.asarray(weights, dtype=np.float32)

                VOCAB_SIZE = weights.shape[0]

                features = {}
                features['word_indices'] = nltk.word_tokenize("hello world")  # ['hello', 'world']
                features['word_indices'] = [word2idx.get(word, UNKNOWN_TOKEN) for word in features['word_indices']]

                glove_weights_initializer = tf.constant_initializer(weights)
                embedding_weights = tf.get_variable(
                    name='embedding_weights',
                    shape=(VOCAB_SIZE, EMBEDDING_DIMENSION),
                    initializer=glove_weights_initializer,
                    trainable=False)
                embedding = tf.nn.embedding_lookup(embedding_weights, features['word_indices'])

                return embedding_weights
