import gc
import os
from pprint import pprint

import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.initializers import RandomUniform
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, classification_report
from tensorflow.models.rnn.ptb.ptb_word_lm import logging
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.models import Model, Input, load_model, model_from_json
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda, Conv1D, MaxPooling1D, \
    Flatten, SpatialDropout1D, Reshape

from processed.Preprocess import load_sentence, labela, flatten
from utilities.setting import TRAIN_, DEV_, TEST_, TEST, DEV, TRAIN, first_set_categories
from utilities.utilities import  vocab_bulid, learn_embedding, pred2label, \
    build_sequences,  true2label
import numpy as np
# from tensorflow.contrib.keras.api.keras.layers import Embedding, Input,  LSTM, Bidirectional, TimeDistributed, SpatialDropout1D, \
# BatchNormalization, Dropout, Dense, Concatenate

from numpy.random import seed

seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

N_CLASSES = len(first_set_categories)


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('Reading corpus data ..')
datasplit, sentences, sent_maxlen, word_maxlen, num_sentence = load_sentence(TRAIN, DEV, TEST)

logger.info('Basic data processing starts ..')
id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv = vocab_bulid(sentences)
word_vocab_size = len(vocb) + 1
char_vocab_size = len(vocb_char) + 1

logger.info('Creating sequence from data ..')
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

###################################################################
#                Word level word representation                   #
###################################################################

logger.info('Setting up model ..')
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


input_text = Input(shape=(105,), dtype="string")
embedding = Lambda(DeepContextualRepresentation, output_shape=(105, 1024))(input_text)

word = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)
word_ = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(word)
word_features = add([word, word_])  # residual connection

out = TimeDistributed(Dense(N_CLASSES, activation="softmax"))(word_features)

model = Model(inputs=[input_text], outputs=out, name='NER_Model')
ada = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)
model.compile(optimizer=ada, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
logger.info('Training initiated. This may take long time to conclude..')
model.fit([np.array(X_train)], y_train, epochs=10, batch_size=75, verbose=1)

gc.collect()
model.save('models/w_only.h5')

logger.info('Initiating prediction of the model')
predict = model.predict([np.array(X_test)], verbose=1, batch_size=75)

logger.info('Printing a sample prediction of the test data')
print(X_test[6])
pprint(np.array(y_test[6]))
print(np.argmax(predict[6], axis=-1))

prediction = np.argmax(predict, axis=-1)

true = flatten(y_test)

logger.info('Building evaluation results')
print(classification_report(np.array(true), np.array(flatten(prediction))))

print(f1_score(true, flatten(prediction), average='weighted'))
