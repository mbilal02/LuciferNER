import tensorflow as tf
import tensorflow_hub as hub
from keras import Input
from keras import backend as K, Model
from keras.initializers import RandomUniform
from keras.layers import Embedding, Reshape, TimeDistributed, Conv1D, MaxPooling1D, regularizers, np, Flatten, Dropout, \
    Bidirectional, LSTM, add, merge, Lambda, Dense, Permute, RepeatVector, concatenate, Activation, dot

# sent_max= 100


def simple_word_level_model(sent_maxlen, dataset_type):
    sess = tf.Session()
    K.set_session(sess)

    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    '''
    Residual bilstm for word-level representation
    '''
    def DeepContextualRepresentation(x):
        '''
        Helper function to create Emlo embedding lookups
        '''
        return elmo_model(inputs={
            "tokens": tf.squeeze(tf.cast(x, "string")),
            "sequence_len": tf.constant(50 * [sent_maxlen])
        },
            signature="tokens",
            as_dict=True)["elmo"]

    input_text = Input(shape=(sent_maxlen,), dtype='string')
    embedding = Lambda(DeepContextualRepresentation, output_shape=(sent_maxlen, 1024))(input_text)
    # hidden layers
    word = Bidirectional(LSTM(units=512, return_sequences=True,
                              recurrent_dropout=0.5, dropout=0.5, kernel_regularizer=regularizers.l2(0.001)))(embedding)
    word_ = Bidirectional(LSTM(units=512, return_sequences=True,
                               recurrent_dropout=0.5, dropout=0.5, kernel_regularizer=regularizers.l2(0.001)))(word)
    word_representations = add([word, word_])  # residual connection

    ner_lstm = Bidirectional(LSTM(units=200, return_sequences=True,
                                  recurrent_dropout=0.3, dropout=0.3, name="Fully_connected"))(word_representations)
    out = TimeDistributed(Dense(dataset_type, activation="softmax"))(ner_lstm)

    model = Model(inputs=[input_text], outputs=out, name='NER_Model')

    return model







