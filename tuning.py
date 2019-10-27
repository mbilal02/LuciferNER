from time import time

from keras import Model, Input
from keras.engine import Layer
from keras.initializers import RandomUniform
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense, merge, Embedding, Lambda, Reshape, Conv1D, \
    GlobalAveragePooling1D, add, regularizers, RepeatVector
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from keras import backend as K, Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from processed.Preprocess import start_build_sequences
from utilities.setting import wnut_b

train_sent, dev_sent, test_sent, X_train, X_dev, X_test, x_c, xc_d, xc_t, y, y_d, \
        y_t, addCharTrain, addCharDev, \
        addCharTest, char_vocab, sent_maxlen, word_maxlen = start_build_sequences(
            vocabulary=wnut_b)
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
            "sequence_len": tf.constant(20 * [105])
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
    conv_net = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'))(c_reshape)
    conv_net = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')) (conv_net)
    conv_net = TimeDistributed(GlobalAveragePooling1D())(conv_net)
    conv_net = TimeDistributed(Dense(32, activation='relu', name='conv_dense')) (conv_net)
    #maxpool_out = TimeDistributed(MaxPooling1D(sent_maxlen))(conv1d_out)
    #char = TimeDistributed(Flatten())(maxpool_out)
    #charOutput = Dropout(0.5)(char)

    return char_input, conv_net

def getResidualBiLSTM(sent_maxlen, units, dropout):
    '''
    Residual bilstm for word-level representation
    '''
    sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # def DeepContextualRepresentation(x):
    #     '''
    #     Helper function to create Emlo embedding lookups
    #     '''
    #     return elmo_model(inputs={
    #         "tokens": tf.squeeze(tf.cast(x, "string")),
    #         "sequence_len": tf.constant(100 * [sent_maxlen])
    #     },
    #         signature="tokens",
    #         as_dict=True)["elmo"]
    input_text = Input(shape=(sent_maxlen,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    word = Bidirectional(LSTM(units=units,
                              return_sequences=True,
                              recurrent_dropout=dropout,
                              dropout=dropout,
                              kernel_regularizer=regularizers.l2(0.001)))(embedding)
    word_ = Bidirectional(LSTM(units=units,
                               return_sequences=True,
                               recurrent_dropout=dropout,
                               dropout=dropout,
                               kernel_regularizer=regularizers.l2(0.001)))(word)
    word_representations = add([word, word_])  # residual connection

    return input_text, word_representations,


def sentence_embedding_encoder():

    def UniversalEmbedding(x):
        session = tf.Session()
        K.set_session(session)
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        embed = hub.Module(module_url)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        return embed(tf.squeeze(tf.cast(x, tf.string)),
                     signature="default", as_dict=True)["default"]
    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(UniversalEmbedding, output_shape=(512,))(input_text)
    #embedding= Reshape([sent_max,512]) (embedding)
    sentence_encoder =Dense(units=256, activation='relu') (embedding)

    return input_text, sentence_encoder

def create_model(units=1, fully_units= 1, dropout=0.0, optimizer = 'rms' ):
    case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                'contains_digit': 6, 'PADDING_TOKEN': 7}
    caseEmbeddings = np.identity(len(case2Idx))
    character_type_input = Input(shape=(sent_maxlen,),
                                 name='character_type_input')
    case = Embedding(input_dim=len(case2Idx),
                     output_dim=caseEmbeddings.shape[1],
                     weights=[caseEmbeddings])(character_type_input)
    char_vocab_size = len(char_vocab) + 1
    #data = dataset_type
    # case_permmute = Permute((2,1)) (case)
    sent_input, sent_out = sentence_embedding_encoder()
    input_char, char_out = getCharCNN(sent_maxlen, word_maxlen, char_vocab_size)
    input_word, word_representations = getResidualBiLSTM(sent_maxlen, units, dropout)
    sent_out = RepeatVector(sent_maxlen)(sent_out)
    wc = merge([word_representations, case, char_out, sent_out],
                   mode='concat',
                   concat_axis=2)
    final_lstm = Bidirectional(LSTM(fully_units,
                                    return_sequences=True,
                                    recurrent_dropout=dropout,
                                    dropout=dropout))(wc)

    out = TimeDistributed(Dense(dataset_type, activation="softmax"))(final_lstm)

    model = Model(inputs=[input_char, input_word, character_type_input, sent_input],
                  outputs=out,
                  name='NER_Model')
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def random_search():
    optimizers = ['rmsprop', 'adam', 'sgd', 'nadam', 'adagrad', 'adadelta', 'adamax']
    #init = ['glorot_uniform', 'normal', 'uniform']
    #epochs = np.array([50, 100, 150])
    #batches = np.array([8, 16, 20, 32, 50])
    dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fully_units = [50, 64, 100, 200, 300, 512, 1024]
    units = [50, 64, 100, 200, 300, 512, 1024]
    param_grid = dict(optimizer=optimizers,
                      dropout=dropout,
                      units= units,
                      fully_units= fully_units)
    start = time()
    model = KerasClassifier(build_fn=create_model, verbose=1)

    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=param_grid,
                                       n_iter=16)
    random_search.fit([np.array(x_c), np.array(X_train), np.array(addCharTrain), np.array(train_sent)], y)
    print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
    means = random_search.cv_results_['mean_test_score']
    stds = random_search.cv_results_['std_test_score']
    params = random_search.cv_results_['params']
    print("total time:", time() - start)
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))