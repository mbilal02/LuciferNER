import gc
import os
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers import RandomUniform
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam

from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, classification_report

from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda, Conv1D, MaxPooling1D, \
    Flatten, SpatialDropout1D, Reshape
from processed.Preprocess import load_sentence, labela, flatten
from utilities.setting import TRAIN_, DEV_, TEST_, TEST, DEV, TRAIN, first_set_categories
from utilities.utilities import vocab_bulid, learn_embedding, pred2label, \
    build_sequences
import numpy as np
import logging

from numpy.random import seed

seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
N_CLASSES = len(first_set_categories)
print(N_CLASSES)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('Reading corpus data')
datasplit, sentences, sent_maxlen, word_maxlen, num_sentence = load_sentence(TRAIN, DEV, TEST)

logger.info('Building vocab')
id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv = vocab_bulid(sentences)


word_vocab_size = len(vocb) + 1
char_vocab_size = len(vocb_char) + 1

logger.info('Setting up input sequences')
x, y, x_c = build_sequences(sentences, vocb_char, labelVoc, word_maxlen, 105)
new_X = []
for seq in x:
    new_seq = []
    for i in range(sent_maxlen):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("__PAD__")
    new_X.append(new_seq)
X = new_X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2018)
X_char_tr, X_char_te, _, _ = train_test_split(x_c, y, test_size=0.25, random_state=2018)

logger.info('Setting up model')
sess = tf.Session()
K.set_session(sess)

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


def DeepContextualRepresentation(x):
    return elmo_model(inputs={
        "tokens": tf.squeeze(tf.cast(x, 'string')),
        "sequence_len": tf.constant(75 * [105])
    },
        signature="tokens",
        as_dict=True)["elmo"]


input_text = Input(shape=(105,), dtype='string')
embedding = Lambda(DeepContextualRepresentation, output_shape=(105, 1024))(input_text)

###################################################################
#                Character level word representation              #
###################################################################

char_in = Input(shape=(sent_maxlen, word_maxlen,))
emb_char = Embedding(input_dim=char_vocab_size, output_dim=30,
                     input_length=(sent_maxlen, word_maxlen,),
                     embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                     name="Character_embedding")(char_in)
dropout = Dropout(0.5)(char_in)
c_reshape = Reshape((sent_maxlen, word_maxlen, 30))(emb_char)
conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30,
                                    padding='same', activation='tanh', strides=1))(c_reshape)
maxpool_out = TimeDistributed(MaxPooling1D(sent_maxlen))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)

###################################################################
#                Word level word representation                   #
###################################################################

word = Bidirectional(LSTM(units=512, return_sequences=True,
                          recurrent_dropout=0.2, dropout=0.2))(embedding)
word_ = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(word)
word_features = add([word, word_])  # residual

###################################################################
#                Concatenating word and character rep             #
###################################################################

concat = concatenate([word_features, char], axis=2)
concat = SpatialDropout1D(0.3)(concat)

###################################################################
#                              NER LSTM                           #
###################################################################

ner_lstm = Bidirectional(LSTM(units=1200, return_sequences=True,
                              recurrent_dropout=0.3, dropout=0.3))(concat)
out = TimeDistributed(Dense(N_CLASSES, activation="softmax"))(ner_lstm)

model = Model(inputs=[input_text, char_in], outputs=out, name='NER_Model')
ada = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)
model.compile(optimizer=ada, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
checkpointer = ModelCheckpoint(filepath='models/w_c_best_model.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

logger.info('Training initiated. This may take long time to conclude..')
model.fit([np.array(X_train), np.array(X_char_tr)], y_train,
          epochs=100, batch_size=75, verbose=1, callbacks=[checkpointer, earlystopper],
          validation_data=([np.array(X_test), np.array(X_char_te)], y_test))

gc.collect()
model.save('Ner_word_character.h5')
predict = model.predict([np.array(X_test), np.array(X_char_te)], verbose=1, batch_size=75)

logger.info('Printing a sample prediction of the test data')
print(X_test[6])
print(np.array(y_test[6]))
print(np.argmax(predict[6], axis=-1))
true = flatten(y_test)
prediction = np.argmax(predict, axis=-1)

logger.info('Building evaluation results')
print(classification_report(np.array(true), np.array(flatten(prediction))))
