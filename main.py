import gc
import os
from keras import metrics
from keras.engine import Model, Input
from keras.initializers import Constant
from keras.layers import Dense, LSTM, Bidirectional, Dropout, K, TimeDistributed, concatenate, SpatialDropout1D, Reshape
from keras.layers import Embedding
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from sklearn.cross_validation import train_test_split
from processed.Preprocess import load_sentence
from utilities.setting import TRAIN_, DEV_, TEST_, TEST, DEV, TRAIN
from utilities.utilities import pad_sequence, vocab_bulid, learn_embedding, pred2label


from numpy.random import seed
seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


EMBEDDING_DIM = 100
n_classes = 21
max_sent = 35
'''
trainData = conllReader('training.conll')
valid = conllReader('dev.conll')
test = conllReader('test.conll')
#merge = tweets_train + tweets_test
train = vocabulary_setup(trainData,valid)
'''
string = [['Shakira', 'B-person'], ['rocked', 'O'], ['Amsterdam', 'B-geo-loc'], ['today', 'O']]
datasplit, sentences, sent_maxlen, word_maxlen, num_sentence = load_sentence(TRAIN_, DEV_, TEST_)
id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv = vocab_bulid(sentences)

word_vocab_size = len(vocb) + 1
char_vocab_size = len(vocb_char) +1

x, y, x_c = pad_sequence(sentences, vocb, vocb_char, labelVoc, word_maxlen =30, sent_maxlen=35)
x_c = x_c.reshape(len(x_c), max_sent * 30)
print(len(x_c))

num_sent = len(sentences)

y = to_categorical(y, num_classes=None)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state= 2018)
X_char_tr, X_char_te, _, _ = train_test_split(x_c, y, test_size=0.1, random_state= 2018)


# Load word Matrix
twitter_embeddings = learn_embedding(vocb, word_vocab_size)


## Word Level Representation

input_shape = max_sent
w_tweet = Input(shape=(input_shape,),)

embed_layer = Embedding(
    input_dim=word_vocab_size,
    output_dim=EMBEDDING_DIM,
    input_length=input_shape,
    embeddings_initializer=Constant(twitter_embeddings),
    trainable=True,
    mask_zero=True,
    name='{}_embed'.format('embedding'))(w_tweet)
#Bug-fixed for Keras embedding layer actually gets frozen to no use :D :D
embed_layer.trainable = False
embed_layer = Dropout(0.5, name='word_dropout')(embed_layer)
embed_layer = Reshape((sent_maxlen, word_maxlen))(embed_layer)
embed_layer = Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.1))(embed_layer)

# Character Level Word Representation#

input_shape_char = max_sent
char_in = Input(shape=(input_shape_char,),)
emb_char = Embedding(input_dim=char_vocab_size, output_dim=30, input_length=(input_shape_char,))(char_in)
emb_char = Dropout(0.3, name='Char_dropout')(emb_char)

# character LSTM to get word encodings by characters (Lample et al 2016)#

char_layer = Bidirectional(LSTM(units=30, return_sequences=True,
                                recurrent_dropout=0.5))(emb_char)

# Concatinating word from bidirectional LSTM and Char LSTM character output layer

concat_layer = concatenate([embed_layer, char_layer], name='concat_layer', axis=2)
concat_layer = SpatialDropout1D(0.3)(concat_layer)
common_dense_layer = Dense(100, name='Dense_concat_layer')(concat_layer)

out = Dense(21, name='final_layer', activation='softmax')(common_dense_layer)

#Crf layer
#crf = CRF(21,sparse_target=True)

#TODO: check if (CRF) -log liklihood makes sense with CRF
#out = crf(cat_output)

model = Model(inputs=[w_tweet,char_in], outputs=out, name='NER_Model')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# early_stopping = EarlyStopping(patience=20, verbose=1)
model.fit([X_train,X_char_tr],y_train, epochs=30, validation_data=[[X_test, X_char_te], y_test], shuffle=True, verbose=1)

model.evaluate([X_test,X_char_te], y_test, verbose=1)
gc.collect()

predict = model.predict([X_test,X_char_te], verbose=1)
test_y_true = y_test[X_test > 0]

print('predicted')
pred_labels = pred2label(predict)
print(pred_labels)
test_labels = pred2label(y_test)
print('true')
print(test_labels)
#print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
#print(classification_report(test_labels, pred_labels))



