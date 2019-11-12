from time import time
import talos as ta

from keras import Input, callbacks
from keras.activations import elu, relu
from keras.engine import Layer
from keras.initializers import RandomUniform
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense, merge, Embedding, Lambda, Reshape, Conv1D, \
    GlobalAveragePooling1D, add, regularizers, RepeatVector
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from keras import backend as K, Model
from keras.optimizers import Adam, Nadam, RMSprop, Adamax, SGD, Adagrad
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe, rand
from talos.model import lr_normalizer

from processed.Preprocess import start_build_sequences
from utilities.setting import wnut_b

train_sent, dev_sent, test_sent, X_train, X_dev, X_test, x_c, xc_d, xc_t, y, y_d, \
y_t, addCharTrain, addCharDev, \
addCharTest, char_vocab, sent_maxlen, word_maxlen = start_build_sequences(vocabulary=wnut_b)
y = y.reshape(y.shape[0], y.shape[1], 1)
y_t = y_t.reshape(y_t.shape[0], y_t.shape[1], 1)
y_d = y_d.reshape(y_d.shape[0], y_d.shape[1], 1)

dataset_type = 13
class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(inputs={
            "tokens": tf.squeeze(tf.cast(x, "string")),
            "sequence_len": tf.constant(100 * [105])
        },
            signature="tokens",
            as_dict=True)["elmo"]
        return result

        # def compute_mask(self, inputs, mask=None):
        # return K.not_equal(inputs, '__PAD__')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 105, self.dimensions)

def getCharCNN(sent_maxlen, word_maxlen, char_vocab_size):
    '''
    Character_level CNN for character representations based on mentioned citation mentioned in the paper, however,
     modified to our case
    '''
    char_out_dim = 30
    char_input = Input(shape=(sent_maxlen, word_maxlen))

    char_embed_layer = Embedding(input_dim=char_vocab_size,
                                 output_dim=char_out_dim,
                                 input_length=(sent_maxlen, word_maxlen,),
                                 embeddings_initializer=RandomUniform(minval=-np.sqrt(3 / char_out_dim),
                                                                      maxval=np.sqrt(3 / char_out_dim)))(char_input)
    # dropout = Dropout(0.5)(char_in)
    c_reshape = Reshape((sent_maxlen, word_maxlen, 30))(char_embed_layer)
    conv_net = TimeDistributed(Conv1D(filters={{choice([16,32,64])}},
                                      kernel_size=3,
                                      activation='relu'))(c_reshape)
    conv_net = TimeDistributed(Conv1D(filters={{choice([16,32,64])}},
                                      kernel_size=3,
                                      activation='relu')) (conv_net)
    conv_net = TimeDistributed(GlobalAveragePooling1D())(conv_net)
    conv_net = TimeDistributed(Dense(units={{choice([50, 100, 150, 200, 256, 300, 512, 1024])}},
                                     activation='relu',
                                     name='conv_dense')) (conv_net)
    #maxpool_out = TimeDistributed(MaxPooling1D(sent_maxlen))(conv1d_out)
    #char = TimeDistributed(Flatten())(maxpool_out)
    #charOutput = Dropout(0.5)(char)

    return char_input, conv_net

def getResidualBiLSTM(sent_maxlen, params):
    '''
    Residual bilstm for word-level representation
    '''
    from tensorflow import Session as session
    from keras import backend as l
    sess = session()
    l.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    input_text = Input(shape=(sent_maxlen,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    word = Bidirectional(LSTM(units=params['first_units'],
                              return_sequences=True,
                              recurrent_dropout=params['dropout'],
                              dropout=params['dropout'],
                              kernel_regularizer=regularizers.l2(0.001)))(embedding)
    word_ = Bidirectional(LSTM(units=params['first_units'],
                               return_sequences=True,
                               recurrent_dropout=params['dropout'],
                               dropout=params['dropout'],
                               kernel_regularizer=regularizers.l2(0.001)))(word)
    word_representations = add([word, word_])  # residual connection

    return input_text, word_representations

def UniversalEmbedding(x):
    session = tf.Session()
    K.set_session(session)
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    embed = hub.Module(module_url)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    return embed(tf.squeeze(tf.cast(x, tf.string)),
                     signature="default", as_dict=True)["default"]
def sentence_embedding_encoder(params):

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(UniversalEmbedding, output_shape=(512,))(input_text)
    sentence_encoder = Dense(units=params['hidden_sent'], activation=params.get('sent_act')) (embedding)

    return input_text, sentence_encoder

def create_model(x_train, y_train,x_val, y_val, params):
    case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                'contains_digit': 6, 'PADDING_TOKEN': 7}
    caseEmbeddings = np.identity(len(case2Idx))
    character_type_input = Input(shape=(sent_maxlen,),
                                 name='character_type_input')
    case = Embedding(input_dim=len(case2Idx),
                     output_dim=caseEmbeddings.shape[1],
                     weights=[caseEmbeddings])(character_type_input)
    char_vocab_size = len(char_vocab) + 1
    input_sent, sent_out = sentence_embedding_encoder(params)

    char_out_dim = 30
    char_input = Input(shape=(sent_maxlen, word_maxlen))

    char_embed_layer = Embedding(input_dim=char_vocab_size,
                                 output_dim=char_out_dim,
                                 input_length=(sent_maxlen, word_maxlen,),
                                 embeddings_initializer=RandomUniform(minval=-np.sqrt(3 / char_out_dim),
                                                                      maxval=np.sqrt(3 / char_out_dim)))(char_input)
    # dropout = Dropout(0.5)(char_in)
    c_reshape = Reshape((sent_maxlen, word_maxlen, 30))(char_embed_layer)
    conv_net = TimeDistributed(Conv1D(filters=params['filters'],
                                      kernel_size=3,
                                      activation='relu'))(c_reshape)
    conv_net = TimeDistributed(Conv1D(filters=params['filters'],
                                      kernel_size=3,
                                      activation='relu'))(conv_net)
    conv_net = TimeDistributed(GlobalAveragePooling1D())(conv_net)
    char_out = TimeDistributed(Dense(units=params.get('hidden_char'),
                                     activation=params.get('char_act'),
                                     name='conv_dense'))(conv_net)
    input_word, word_representations = getResidualBiLSTM(sent_maxlen, params)
    sent_out = RepeatVector(sent_maxlen)(sent_out)
    wc = merge([char_out, word_representations, case, sent_out],
                   mode='concat',
                   concat_axis=2)
    final_lstm = Bidirectional(LSTM(params['last_units'],
                                    return_sequences=True,
                                    recurrent_dropout=params['dropout'],
                                    dropout=params['dropout']))(wc)

    out = TimeDistributed(Dense(dataset_type, activation="softmax"))(final_lstm)

    model = Model(inputs=[char_input, input_word, character_type_input, input_sent],
                  outputs=out,
                  name='NER_Model')
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                            patience=5, min_lr=0.0001)
    model.compile(optimizer=params['optimizers'](lr=lr_normalizer(params['lr'],params['optimizers'])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', ta.utils.metrics.f1score])

    model.summary()
    model.fit(x_train, y_train,
              epochs=100,
              batch_size=100,
              verbose=1,
              callbacks=[reduce_lr],
              validation_data=(x_val, y_val),
              shuffle=True)



if __name__ == '__main__':
    param = {'last_units':[50, 64, 100, 200, 300, 512, 1024],
             'first_units': [50, 64, 100, 200, 300, 512, 1024],
             'hidden_sent': [50, 64, 100, 200, 256, 300, 512, 1024],
             'hidden_char': [50, 64, 100, 200, 256, 300, 512, 1024],
             'dropout':[0,0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7],
             'optimizers':[Adam, Nadam, RMSprop, Adamax, SGD, Adagrad],
             'filters':[8, 16, 32, 64, 128],
             'lr':[0.0001,0.001,0.01,0.1,0.19,0.5],
             'sent_act': [relu, elu],
             'char_act': [relu, elu]}
    start = time()
    t = ta.Scan(x=[np.array(x_c),
                                np.array(X_train),
                                np.array(addCharTrain),
                   np.array(train_sent)],
                y=y,
                params=param,
                model=create_model
                , x_val = [np.array(xc_d),
                                np.array(X_dev),
                                np.array(addCharDev),
                           np.array(dev_sent)]
                , y_val = y_d
                , experiment_name = 'optimization'
                , round_limit = 10
               )


    print("Process finished in ET:", time() - start)