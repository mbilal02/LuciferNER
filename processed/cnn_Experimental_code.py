from keras import Input
from keras.initializers import RandomUniform
from keras.layers import Embedding, Dropout, Conv1D, GlobalAveragePooling1D, Dense, GlobalMaxPooling1D

import numpy as np


def _get_input_layer(shape, name):
    return Input(shape=shape, dtype='int32', name='{}_input'.format(name))


def _rand_unif_emb_layer(input_layer, input_dim, output_dim,
                         input_len, name, seed=1337):
    uniform = RandomUniform(seed=seed,
                            minval=-np.sqrt(3 / output_dim),
                            maxval=np.sqrt(3 / output_dim))
    embed_layer = Embedding(input_dim=input_dim,
                            output_dim=output_dim,
                            input_length=input_len,
                            embeddings_initializer=uniform,
                            trainable=False,
                            name='{}_embed'.format(name))(input_layer)
    embed_layer = Dropout(0.5, name='{}_embed_dropout'.format(name))(embed_layer)
    return embed_layer


def add_conv_layers(embedded, name, filters=64, kernel_size=3, dense_units=32, convs=2):
    conv_net = embedded
    for _ in range(convs):
        conv_net = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(conv_net)
    conv_net = GlobalMaxPooling1D()(conv_net)
    conv_net = Dense(dense_units, activation='relu', name='{}_dense'.format(name))(conv_net)
    return conv_net


def get_char_cnn(char_max_len,
                 char_vocab_size,
                 char_dim=30,
                 name='char_layer'):
    char_input = _get_input_layer((char_max_len,), name)
    char_embed = _rand_unif_emb_layer(char_input, char_vocab_size, char_dim, char_max_len, name)
    char_encoded = add_conv_layers(char_embed, name + '_encoded')
    return [char_input], char_encoded
