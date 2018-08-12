from pprint import pprint

from keras import Input
from keras.layers import Dense, LSTM, Bidirectional, concatenate, Reshape, TimeDistributed, Convolution1D, MaxPooling1D, \
    Dropout, BatchNormalization, Activation
from keras.layers import Embedding
from keras.layers import Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import asarray
from numpy import zeros
from feature_extraction.Preprocess import read_datasets, encode_doc, read_test_datasets, read_file_as_lists, \
    encode_orthography
import os

from feature_extraction.cnn_Experimental_code import get_char_cnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(tweets_train, labels_train), (tweets_test, labels_test) = read_datasets("../data/training.conll")
# (tweets_train1, labels_train1), (tweets_test1, labels_test1) = read_test_datasets("../data/dev.conll")
tweet_test, label_test = read_file_as_lists("../data/dev.conll", delimiter='\t')
EMBEDDING_DIM = 200

train_tweet_idx = encode_doc(tweets_train)
train_label_idx = encode_doc(labels_train)
test_tweet = encode_doc(tweet_test)
test_labels = encode_doc(label_test)

# character Setting
index2ortho = ['x', 'c', 'C', 'n', 'p']
ortho_dim = 30
ortho_max_length = 20
x_ortho_train = encode_orthography(tweets_train, ortho_max_length)
x_ortho_test = encode_orthography(tweets_test, ortho_max_length)

# prepare tokenizer

t = Tokenizer()

t.fit_on_texts(tweets_train)
vocab_size = len(t.word_index) + 1
'''# integer encode the documents
encoded_docs = t.texts_to_sequences(tweets_train)
print(encoded_docs)
'''
# pad documents to a max length of 4 words
max_length = 4
x_train = pad_sequences(train_tweet_idx, maxlen=max_length, padding='post')
x_test = pad_sequences(test_tweet, maxlen=max_length, padding='post')

print(x_train)

# load the whole embedding into memory
embeddings_index = dict()
f = open('../data/glove.twitter.27B.200d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, EMBEDDING_DIM))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
pprint(embedding_matrix.shape)

# create the model
# embedding_vecor_length = 32
word_maxlen = 30
sent_maxlen = 35
char_dim = 30
w_emb_dim_char_level = 50
# model = Sequential()
# e = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=4, trainable=False, mask_zero=False)
# model.add(Embedding(vocab_size, embedding_vecor_length, input_length=4))
w_tweet = Input(shape=(sent_maxlen,), dtype='int32')
w_emb = Embedding(input_dim=vocab_size, output_dim=200, weights=[embedding_matrix], input_length=sent_maxlen,
                  mask_zero=True)(
    w_tweet)
w_feature = Bidirectional(LSTM(200, return_sequences=True, input_shape=(4,)))(w_emb)

# char_inputs, char_encoded = get_char_cnn(ortho_max_length, len(index2ortho), ortho_dim, 'char_ortho')

# char level word representation
c_tweet = Input(shape=(sent_maxlen * word_maxlen,), dtype='int32')
c_emb = Embedding(input_dim=ortho_max_length, output_dim=char_dim, input_length=sent_maxlen * word_maxlen,
                  mask_zero=False)(
    c_tweet)
c_reshape = Reshape((sent_maxlen, word_maxlen, char_dim))(c_emb)
c_conv1 = TimeDistributed(Convolution1D(nb_filter=32, filter_length=2, border_mode='same', activation='relu'))(
    c_reshape)
c_pool1 = TimeDistributed(MaxPooling1D(pool_length=2))(c_conv1)
c_dropout1 = TimeDistributed(Dropout(0.25))(c_pool1)
c_conv2 = TimeDistributed(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))(
    c_dropout1)
c_pool2 = TimeDistributed(MaxPooling1D(pool_length=2))(c_conv2)
c_dropout2 = TimeDistributed(Dropout(0.25))(c_pool2)
c_conv3 = TimeDistributed(Convolution1D(nb_filter=32, filter_length=4, border_mode='same', activation='relu'))(
    c_dropout2)
c_pool3 = TimeDistributed(MaxPooling1D(pool_length=2))(c_conv3)
c_dropout3 = TimeDistributed(Dropout(0.25))(c_pool3)
c_batchNorm = BatchNormalization()(c_dropout3)
c_flatten = TimeDistributed(Flatten())(c_batchNorm)
c_fullConnect = TimeDistributed(Dense(100))(c_flatten)
c_activate = TimeDistributed(Activation('relu'))(c_fullConnect)
c_emb2 = TimeDistributed(Dropout(0.25))(c_activate)
c_feature = TimeDistributed(Dense(w_emb_dim_char_level))(c_emb2)

merge_w_c_emb = concatenate([w_feature, c_feature], name='concat_layer')
w_c_feature = Bidirectional(LSTM(output_dim=200, return_sequences=True))(merge_w_c_emb)
print(merge_w_c_emb)
print(w_c_feature)
