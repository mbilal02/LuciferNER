
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


def learn_embedding(vocab):
    embeddings_index = dict()
    b = 0
    f = open('data/glove.twitter.27B.200d.txt', 'r', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = numpy.zeros((len(vocab) + 1, 200))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

            return embedding_matrix

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


def build_sequences(sentences, vocab_char, labelVoc, word_maxlen, sent_maxlen):
    '''
        This function is used to pad the word into the same length.

    '''
    x = []
    x_w = []
    y = []
    for sentence in sentences:
        w_id = []
        w = []
        y_id = []
        for word_label in sentence:
            w_id.append(word_label[0])
            #w.append(vocabulary[word_label[0]])

            y_id.append(labelVoc[word_label[1]])
        x.append(w_id)
        #x_w.append(w)
        y.append(y_id)

    y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=sent_maxlen, padding="post")
    #x_word = tf.keras.preprocessing.sequence.pad_sequences(x_w, maxlen=sent_maxlen, padding="post")

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
            p_i = np.argmax(p, axis=-1)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out

def true2label(pred,labelVoc):
    idx2tag = {i: w for w, i in labelVoc.items()}
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i= np.argmax(p)
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


# Method to compute the accruarcy. Call predict_labels to get the labels for the dataset
def compute_f1(predictions, correct, idx2Label):
    label_pred = []
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])

    label_correct = []
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])

    # print label_pred
    # print label_correct

    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);

    return prec, rec, f1


def compute_precision(guessed_sentences, correct_sentences):
    assert (len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert (len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B':  # A new chunk starts
                count += 1

                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True

                    while idx < len(guessed) and guessed[idx][0] == 'I':  # Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False

                        idx += 1

                    if idx < len(guessed):
                        if correct[idx][0] == 'I':  # The chunk in correct was longer
                            correctlyFound = False

                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:
                idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision
