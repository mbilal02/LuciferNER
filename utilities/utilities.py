
from collections import Counter

import gensim
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras_preprocessing import image
from keras import preprocessing
from keras import backend as K
import keras as keras
from keras_preprocessing.sequence import pad_sequences




######################################################
#   Word Matrix, word and character vocabulary build #
######################################################

def load_word_matrix(vocabulary, size=200):
    '''
        This function is used to convert words into word vectors
    '''

    b = 0
    word_matrix = np.zeros((len(vocabulary) + 1, size))
    model = gensim.models.KeyedVectors.load_word2vec_format("../data/glove.txt", binary=False, encoding='utf8')
    for word, i in vocabulary.items():
        try:
            word_matrix[i] = model[word.lower().encode('utf8')]
            # index2word, embeddings = pick_embeddings_by_indexes(get_uniq_elems(corpus), model.vocab.item)
        except KeyError:
            # if a word is not include in the vocabulary, it's word embedding will be set by random.
            word_matrix[i] = np.random.uniform(-0.25, 0.25, size)
            b += 1
    print('there are %d words not in model' % b)
    return word_matrix


index2category = [
    'B-corporation',
    'B-creative-work',
    'B-group',
    'B-location',
    'B-person',
    'B-product',
    'I-corporation',
    'I-creative-work',
    'I-group',
    'I-location',
    'I-person',
    'I-product',
    'O'
]


def vocab_bulid(sentences):
    '''
    input:
        sentences list,

    output:
        VOcabulary

    '''
    words = []
    chars = []
    labels = []

    for sentence in sentences:
        for word_label in sentence:
            words.append(word_label[0])
            labels.append(word_label[1])
            for char in word_label[0]:
                chars.append(char)
    word_counts = Counter(words)
    vocb_inv = [x[0] for x in word_counts.most_common()]
    vocb = {x: i + 1 for i, x in enumerate(vocb_inv)}
    vocb['PAD'] = 0
    id_to_vocb = {i: x for x, i in vocb.items()}

    char_counts = Counter(chars)
    vocb_inv_char = [x[0] for x in char_counts.most_common()]
    vocb_char = {x: i + 1 for i, x in enumerate(vocb_inv_char)}

    labels_counts = Counter(labels)

    labelVoc_inv, labelVoc = label_index(labels_counts)

    return [id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv]


def pad_sequence(sentences, vocabulary, labelVoc, sent_maxlen=35):
    '''
        This function is used to pad the word into the same length.

    '''
    x = []
    y = []
    for sentence in sentences:
        w_id = []
        y_id = []
        for word_label in sentence:
            w_id.append(vocabulary[word_label[0]])
            y_id.append(labelVoc[word_label[1]])
        x.append(w_id)
        y.append(y_id)

    y = pad_sequences(y, maxlen=sent_maxlen).astype(np.int32)
    x = pad_sequences(x, maxlen=sent_maxlen).astype(np.int32)

    x = np.asarray(x)
    y = np.asarray(y)

    return [x, y]


'''
def label_index(labels_counts):
	"""
	   the input is the output of Counter. This function defines the (label, index) pair,
	   and it cast our datasets label to the definition (label, index) pair.
	"""

	num_labels = len(labels_counts)
	labelVoc_inv = [x[0] for x in labels_counts.most_common()]

	labelVoc = { 'B-corporation':12,
    'B-creative-work':11,
    'B-group':10,
    'B-location':9,
    'B-person':8,
    'B-product':7,
    'I-corporation':6,
    'I-creative-work':5,
    'I-group':4,
    'I-location':3,
    'I-person':2,
    'I-product':1,
    'O':0}
	if len(labelVoc) < num_labels:
		for key,value in labels_counts.items():
			if not key in labelVoc:
				labelVoc.setdefault(key, len(labelVoc))
	return labelVoc_inv, labelVoc
'''


def label_index(labels_counts):
    '''
        defining (label, index) pair.
    '''

    num_labels = len(labels_counts)
    labelVoc_inv = [x[0] for x in labels_counts.most_common()]

    labelVoc = {
        'I-tvshow': 20,
        'B-tvshow': 19,
        'I-sportsteam': 18,
        'B-sportsteam': 17,
        'I-product': 16,
        'B-product': 15,
        'I-person': 14,
        'B-person': 13,
        'B-other': 12,
        'I-other': 11,
        'I-musicartist': 10,
        'B-musicartist': 9,
        'I-movie': 8,
        'B-movie': 7,
        'I-geo-loc': 6,
        'B-geo-loc': 5,
        'I-facility': 4,
        'B-facility': 3,
        'I-company': 2,
        'B-company': 1,
        'O': 0}
    if len(labelVoc) < num_labels:
        for key, value in labels_counts.items():
            if not key in labelVoc:
                labelVoc.setdefault(key, len(labelVoc))
    return labelVoc_inv, labelVoc






######################################################
#   Metric Function , Training history plotting      #
######################################################



def show_training_loss_plot(hist):

    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    plt.plot(range(len(train_loss)), train_loss, color="red", label="Train Loss")
    plt.plot(range(len(train_loss)), val_loss, color="blue", label="Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.show()


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    precision = c1 / c2

    recall = c1 / c3

    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score


def decode_predictions(predictions, idx2label):
    return [idx2label[pred] for pred in predictions]


def f1(label, prediction):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(0, label.shape[0]):
        if prediction[i] == 1:
            if prediction[i] == label[i]:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if prediction[i] == label[i]:
                true_negatives += 1
            else:
                false_negatives += 1

    accuracy = (true_positives + true_negatives) \
               / (true_positives + true_negatives + false_positives + false_negatives)

    precision = true_positives / (true_positives + false_positives)

    recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 / ((1 / precision) + (1 / recall))

    return accuracy, precision, recall, f1_score

