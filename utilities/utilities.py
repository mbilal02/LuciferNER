
from collections import Counter

import gensim
import numpy
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf



######################################################
#   Word Matrix, word and character vocabulary build #
######################################################



def learn_embedding(vocab, word_vocab_size):
    embeddings_index = dict()
    b = 0
    f = open('data/glove.twitter.27B.100d.txt', 'r', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((word_vocab_size, 100))
    for word, index in vocab.items():
        if index > word_vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            else:
                # if a word is not include in the vocabulary, it's word embedding will be set by random.
                embedding_matrix[index] = np.random.uniform(-0.25, 0.25, 100)
                b += 1
            print('there are %d words not in model' % b)

            return embedding_matrix


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


def pad_sequence(sentences, vocabulary, vocab_char, labelVoc, word_maxlen=30, sent_maxlen=35):
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

    y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=sent_maxlen, padding="post", value=labelVoc["O"])
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=sent_maxlen, padding="post")
    x_c = []
    for sentence in sentences:
        s_pad = np.zeros([sent_maxlen, word_maxlen], dtype=np.int32)
        s_c_pad = []
        for word_label in sentence:
            w_c = []
            char_pad = np.zeros([word_maxlen], dtype=np.int32)
            for char in word_label[0]:
                w_c.append(vocab_char[char])
            if len(w_c) <= word_maxlen:
                char_pad[:len(w_c)] = w_c
            else:
                char_pad = w_c[:word_maxlen]

            s_c_pad.append(char_pad)

        for i in range(len(s_c_pad)):
            s_pad[sent_maxlen - len(s_c_pad) + i, :len(s_c_pad[i])] = s_c_pad[i]
        x_c.append(s_pad)

    x_c = np.asarray(x_c)
    x = np.asarray(x)
    y = np.asarray(y)

    return [x, y, x_c]


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


# Decoding labels

def pred2label(pred,labelVoc):
    idx2tag = {i: w for w, i in labelVoc.items()}
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out


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


def classification_report(y_true, y_pred, labels):
    y_true = numpy.asarray(y_true).ravel()
    y_pred = numpy.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('', 'recall', 'precision', 'f1-score', 'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report])
    N = len(y_true)
    print(formatter('avg / total', sum(report2[0]) / N, sum(report2[1]) / N, sum(report2[2]) / N, N) + '\n')


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

