import tensorflow as tf
import tensorflow_hub as hub
from keras import Input
from keras import backend as K, Model
from keras.initializers import RandomUniform
from keras.layers import Embedding, Reshape, TimeDistributed, Conv1D, MaxPooling1D, regularizers, np, Flatten, Dropout, \
    Bidirectional, LSTM, add, merge, Lambda, Dense, Permute, RepeatVector, concatenate, Activation, dot

sess = tf.Session()
K.set_session(sess)

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
sent_max= 100

def DeepContextualRepresentation(x):
    '''
    Helper function to create Emlo embedding lookups
    '''
    return elmo_model(inputs={
        "tokens": tf.squeeze(tf.cast(x, 'string')),
        "sequence_len": tf.constant(50 * [sent_max])
    },
        signature="tokens",
        as_dict=True)["elmo"]


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
    conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30,
                                        padding='same', activation='tanh', strides=1,
                                        kernel_regularizer=regularizers.l2(0.001)))(c_reshape)
    maxpool_out = TimeDistributed(MaxPooling1D(sent_maxlen))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    charOutput = Dropout(0.5)(char)

    return char_input, charOutput


def getResidualBiLSTM(sent_maxlen):

    '''
    Residual bilstm for word-level representation
    '''
    input_text = Input(shape=(sent_maxlen,), dtype='string')
    embedding = Lambda(DeepContextualRepresentation, output_shape=(sent_maxlen, 1024))(input_text)
    word = Bidirectional(LSTM(units=512, return_sequences=True,
                              recurrent_dropout=0.5, dropout=0.5, kernel_regularizer=regularizers.l2(0.001)))(embedding)
    word_ = Bidirectional(LSTM(units=512, return_sequences=True,
                               recurrent_dropout=0.5, dropout=0.5, kernel_regularizer=regularizers.l2(0.001)))(word)
    word_representations = add([word, word_])  # residual connection

    return input_text, word_representations,


def build_bilstm_cnn_model(sent_maxlen, word_maxlen, char_vocab, dataset_type):

    char_vocab_size = len(char_vocab) + 1

    input_char, char_out = getCharCNN(sent_maxlen, word_maxlen, char_vocab_size)
    input_word, word_representations = getResidualBiLSTM(sent_maxlen)

    concat = merge([char_out, word_representations], mode='concat',
                   concat_axis=2)  # Residual and Highway connections are concatenated
    ner_lstm = Bidirectional(LSTM(units=200, return_sequences=True,
                                  recurrent_dropout=0.3, dropout=0.3))(concat)
    out = TimeDistributed(Dense(dataset_type, activation="softmax"))(ner_lstm)

    model = Model(inputs=[input_word, input_char], outputs=out, name='NER_Model')

    return model


def addtTextGuidedVisualAttention(sent_maxlen, text_rep):

    '''
    Helper function to build text guided visual attention as cited in the paper
    '''
    img_input = Input(shape=(1, 7, 7, 512), name='Image_input')
    img_reshape = Reshape((512, 7 * 7))(img_input)

    # img_reshape = Flatten() (img_reshape)
    # img_res = Reshape((512)) (img_reshape)
    img_permute = Permute((2, 1), name='image_reshape1')(img_reshape)
    print(img_permute.shape)
    img_permute_reshape = TimeDistributed(RepeatVector(sent_maxlen), name='Repeat_maxlen')(img_permute)
    img_permute_ = Permute((2, 1, 3), name='image_reshape2')(img_permute_reshape)
    # print(img_permute_.shape())
    text_repeat_vector = TimeDistributed(RepeatVector(7 * 7), name='repeat_image_w_h')(text_rep)
    text_repeat_vector = TimeDistributed(TimeDistributed(Dense(300)))(text_repeat_vector)
    img_permute_ = TimeDistributed(TimeDistributed(Dense(300)))(img_permute_)

    merge_f = concatenate([text_repeat_vector, img_permute_])

    att_w = TimeDistributed(Activation('tanh'))(merge_f)
    att_w = TimeDistributed(Dense(1))(att_w)
    att_w = TimeDistributed(Flatten())(att_w)
    att_w_probability = TimeDistributed(Activation('softmax'), name='visual_attention_weights')(att_w)

    output_visual_attention = dot([att_w_probability, img_permute_], axes=(2, 2))

    return output_visual_attention, img_input


def addTextAttention(text_representation, sent_maxlen):
    '''
    Helper fuction to compute text attention as described in the paper.
    '''
    tweet_dense = TimeDistributed(Dense(300))(text_representation)
    # tweet_flatten = Flatten()(tweet_dense)
    tweet_rep = TimeDistributed(RepeatVector(sent_maxlen))(tweet_dense)
    tweet_rep = Reshape((sent_maxlen, sent_maxlen, 300,))(tweet_rep)
    att_w_t = TimeDistributed(Activation('tanh'))(tweet_rep)
    att_w_t = TimeDistributed(Dense(1))(att_w_t)
    att_w_t = TimeDistributed(Flatten())(att_w_t)
    att_w_probability = TimeDistributed(Activation('softmax'), name='text_attention_weights')(att_w_t)

    output_textual_attention = dot([att_w_probability, tweet_rep], axes=2)

    return output_textual_attention


def build_multimodal_attnetion_network(sent_maxlen, word_maxlen, char_vocab, dataset_type):

    char_vocab_size = len(char_vocab) + 1

    input_char, char_out = getCharCNN(sent_maxlen, word_maxlen, char_vocab_size)
    input_word, word_representations = getResidualBiLSTM(sent_maxlen)

    text_rep = merge([char_out, word_representations], mode='concat', concat_axis=2)

    # process text features with a BiLSTM layer to extract final representation

    final_text_representation = (Bidirectional(LSTM(units=300, return_sequences=True, recurrent_dropout=0.3,
                                                    dropout=0.3, name='Text_feature_BLSTM')))(text_rep)

    visual_attention, input_image = addtTextGuidedVisualAttention(sent_maxlen, text_rep)

    text_attention = addTextAttention(final_text_representation, sent_maxlen)

    sum = getVectorSum(text_attention, visual_attention)

    ner_lstm = Bidirectional(LSTM(units=200, return_sequences=True,
                                  recurrent_dropout=0.3, dropout=0.3))(sum)
    out = TimeDistributed(Dense(dataset_type, activation="softmax"))(ner_lstm)

    model = Model(inputs=[input_word, input_char, input_image], outputs=out, name='NER_Model')

    return model


def getVectorSum(a, b):
    '''
    Helper function to take sum of two tensors
    '''
    return merge([a, b], mode='sum')
