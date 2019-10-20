import csv
from collections import Counter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy
import numpy as np
import tensorflow as tf


######################################################
#   Word Matrix and vocabulary build                 #
######################################################
def flatten(list):
    '''
    Helper function that flattens 2D lists.
    '''
    return [i for sublist in list for i in sublist]


wnut_b = {'B-corporation': 12,
          'B-creative-work': 11,
          'B-group': 10,
          'B-location': 9,
          'B-person': 8,
          'B-product': 7,
          'I-corporation': 6,
          'I-creative-work': 5,
          'I-group': 4,
          'I-location': 3,
          'I-person': 2,
          'I-product': 1,
          'O': 0}

def learn_embedding(vocab):
    '''
    Creating Global vectors weight matrix, later might be used for initializing the word embeddings layer
    '''
    embeddings_index = dict()
    file = open('data/glove.twitter.27B.200d.txt', 'r', encoding='utf8')
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    file.close()

    embedding_matrix = numpy.zeros((len(vocab) + 1, 200))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

            return embedding_matrix


def create_lookup(sentences, voc):
    '''
    Creates character and label lookups
    '''

    # You do not need to worry about character lookups and other stuff at the moment
    words = []
    chars = []
    labels = []

    for sentence in sentences:
        for word_label in sentence:
            words.append(word_label[0])
            #only this is important list of list of (word, label) pair 0 index word and 1 index label
            labels.append(word_label[1])

    labels_counts = Counter(labels)

    labelVoc_inv, labelVoc = label_index(labels_counts, voc)

    return labelVoc

def create_sequences(sentences, labelVoc, word_maxlen, sent_maxlen):
    '''
        This function is used to pad the word into the same length.
    '''
    x = []
    y = []

    for sentence in sentences:
        w_id = []
        y_id = []
        for word_label in sentence:
            w_id.append(word_label[0])
            y_id.append(labelVoc[word_label[1]])
        x.append(w_id)
        y.append(y_id)
    y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=sent_maxlen, padding="post")
    return [x, y, sent_maxlen, word_maxlen]


def label_index(labels_counts, labelVoc):
    '''
       the input is the output of Counter. This function defines the (label, index) pair,
       and it cast our datasets label to the definition (label, index) pair.
    '''

    num_labels = len(labels_counts)
    labelVoc_inv = [x[0] for x in labels_counts.most_common()]
    if len(labelVoc) < num_labels:
        for key, value in labels_counts.items():
            if not key in labelVoc:
                labelVoc.setdefault(key, len(labelVoc))
    return labelVoc_inv, labelVoc

######################################################
#   Prediction files prepration                      #
######################################################
def write_file(filename, dataset, delimiter='\t'):
    """dataset is a list of tweets where each token can be a tuple of n elements"""
    with open(filename, '+w', encoding='utf8') as stream:
        writer = csv.writer(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE, quotechar='')

        for tweet in dataset:
            writer.writerow(list(tweet))


def save_predictions(filename, tweets, labels, predictions):
    """save a file with token, label and prediction in each row"""
    dataset, i = [], 0
    for n, tweet in enumerate(tweets):
        tweet_data = list(zip(tweet, labels[n], predictions[n]))
        dataset += tweet_data + [()]
    write_file(filename, dataset)


def getLabels(y_test, vocabulary):
    '''
    Maps integer to the label map
    '''
    #
    classes = []
    # y = np.array(y_test).tolist()
    for i in y_test:
        label = []
        pre = [[k for k, v in vocabulary.items() if v == j] for j in i]
        for i in pre:
            for j in i:
                label.append(j)
        classes.append(label)
    return classes




######################################################
#   Training history plotting                        #
######################################################



def show_training_loss_plot(hist):
    '''
    Graph plotter
    '''

    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    plt.plot(range(len(train_loss)), train_loss, color="red", label="Train Loss")
    plt.plot(range(len(train_loss)), val_loss, color="blue", label="Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.show()
