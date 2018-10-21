import gc
import os
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers import RandomUniform
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam, RMSprop, Nadam

from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda, Conv1D, MaxPooling1D, \
    Flatten, SpatialDropout1D, Reshape, GlobalAveragePooling1D
from processed.Preprocess import load_sentence, labela, flatten, conllReader
from utilities.setting import TRAIN_, DEV_, TEST_, TEST, DEV, TRAIN, first_set_categories, TEST_15
from utilities.utilities import vocab_bulid, learn_embedding, pred2label, \
    build_sequences,compute_f1
import numpy as np
import logging
from keras import regularizers
from numpy.random import seed

seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
N_CLASSES = 21
print(N_CLASSES)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('Reading corpus data')
datasplit, sentences, sent_maxlen, word_maxlen, num_sentence = load_sentence(TRAIN_, DEV_, TEST_)

logger.info('Building vocab')
id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv = vocab_bulid(sentences)


word_vocab_size = len(vocb) + 1
char_vocab_size = len(vocb_char) + 1
train_sent= conllReader(TRAIN_)
dev_sent= conllReader(DEV_)

test_sents = conllReader(TEST_)
training_sents = (train_sent + dev_sent)
logger.info('Setting up input sequences')
x, y, x_c = build_sequences(training_sents, vocb_char, labelVoc, word_maxlen, sent_maxlen)
x_t, y_t , xc_t = build_sequences(test_sents, vocb_char, labelVoc, word_maxlen, sent_maxlen)
def sequence_helper(x_in):
    new_X = []
    for seq in x_in:
        new_seq = []
        for i in range(sent_maxlen):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_X.append(new_seq)
    return new_X


X_train = sequence_helper(x)
X_test = sequence_helper(x_t)
print(len(X_train))
print(len(X_test))
y_train= y
y_test= y_t
X_char_tr = x_c
X_char_te = xc_t


logger.info('Setting up model')
sess = tf.Session()
K.set_session(sess)

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


def DeepContextualRepresentation(x):
    return elmo_model(inputs={
        "tokens": tf.squeeze(tf.cast(x, 'string')),
        "sequence_len": tf.constant(50 * [sent_maxlen])
    },
        signature="tokens",
        as_dict=True)["elmo"]


input_text = Input(shape=(sent_maxlen,), dtype='string')
embedding = Lambda(DeepContextualRepresentation, output_shape=(sent_maxlen, 1024))(input_text)

###################################################################
#                Character level and word level representation    #
###################################################################

char_out_dim = 30
char_input = Input(shape= (sent_maxlen,word_maxlen))

char_embed_layer = Embedding(input_dim=char_vocab_size,
                            output_dim=char_out_dim,
                            input_length=(sent_maxlen, word_maxlen,),
                            embeddings_initializer=RandomUniform(minval=-np.sqrt( 3 / char_out_dim ),
                                                                 maxval=np.sqrt( 3 / char_out_dim )))(char_input)
#dropout = Dropout(0.5)(char_in)
c_reshape = Reshape((sent_maxlen, word_maxlen, 30))(char_embed_layer)
conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30,
                                    padding='same', activation='tanh', strides=1, kernel_regularizer=regularizers.l2(0.001)))(c_reshape)
maxpool_out = TimeDistributed(MaxPooling1D(sent_maxlen))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)
word = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.5, dropout=0.5,kernel_regularizer=regularizers.l2(0.001)))(embedding)
word_ = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.5, dropout=0.5, kernel_regularizer=regularizers.l2(0.001)))(word)
word_features = add([word, word_])  # residual connection

concat = concatenate([char, word_features], axis=2)
concat = SpatialDropout1D(0.5)(concat)
###################################################################
#                              Final NER LSTM                     #
###################################################################

ner_lstm = Bidirectional(LSTM(units=300, return_sequences=True,
                              recurrent_dropout=0.5, dropout=0.5))(concat)
out = TimeDistributed(Dense(N_CLASSES, activation="softmax"))(ner_lstm)

model = Model(inputs=[char_input, input_text], outputs=out, name='NER_Model')
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=nadam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
checkpointer = ModelCheckpoint(filepath='models/w_c_best_model2017.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

logger.info('Training initiated. This may take long time to conclude..')
model.fit([ np.array(X_char_tr), np.array(X_train)], y_train,
          epochs=18, batch_size=50, verbose=1, callbacks=[checkpointer, earlystopper],
          validation_data=([np.array(X_char_te),np.array(X_test)], y_test), shuffle=True)

gc.collect()
#model.save('Ner_word_character.h5')
predict = model.predict([np.array(X_char_te),np.array(X_test)], verbose=1, batch_size=50)

logger.info('Printing a sample prediction of the test data')
print(X_test[6])
print(np.array(y_test[6]))
print(np.argmax(predict[6], axis=-1))
true = flatten(y_test)
print(X_test[16])
print(np.array(y_test[16]))
print(np.argmax(predict[16], axis=-1))
truth = flatten(y_test)
prediction = np.argmax(predict, axis=-1)

logger.info('Building evaluation results')
print(classification_report(np.array(truth), np.array(flatten(prediction))))
print(confusion_matrix(np.array(truth), np.array(flatten(prediction)), labels=None, sample_weight=None))
