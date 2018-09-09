import gc
import os

from keras.engine import Model, Input
from keras.layers import Dense, LSTM, Bidirectional, Dropout, K
from keras.layers import Embedding
from keras.utils import to_categorical
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from processed.Preprocess import load_sentence
from utilities.setting import TRAIN_, DEV_, TEST_
from utilities.utilities import load_word_matrix, pad_sequence, vocab_bulid

# from numpy.random import seed
# seed(7)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EMBEDDING_DIM = 200
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

print(vocb['rainy'])
print(vocb_inv[3651])
print(id_to_vocb)
print(labelVoc['B-company'])
print(labelVoc_inv)
word_vocab_size = len(vocb) + 1
x, y = pad_sequence(sentences, vocb, labelVoc, sent_maxlen=35)

num_sent = len(sentences)
# Reshaping y in sentence length in to max sentence length
y = y.reshape((num_sent * max_sent))

y = to_categorical(y, num_classes=None)

y = y.reshape((num_sent, max_sent, len(labelVoc)))
print(y.shape)
print(x.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

twitter_embeddings = load_word_matrix(vocb)
# print(y_cat_train)
# print(y_cat_test)
input_shape = max_sent
w_tweet = Input(shape=(input_shape,), dtype='int32')
# w_emb = Embedding(input_dim=twitter_embeddings.shape, output_dim=EMBEDDING_DIM,weights=[twitter_embeddings], input_length=input_shape, mask_zero=False)(
embed_layer = Embedding(
    input_dim=word_vocab_size,
    output_dim=EMBEDDING_DIM,
    input_length=input_shape,
    weights=[twitter_embeddings],
    trainable=False,
    name='{}_embed'.format('embedding'))(w_tweet)
embed_layer = Dropout(0.5, name='{}_embed_dropout'.format('drop'))(embed_layer)

embed_layer = Bidirectional(LSTM(200, batch_size=150, activation='sigmoid', return_sequences=True))(embed_layer)
# embed_layer = Bidirectional(LSTM(200,  batch_size=150, return_sequences=False))(embed_layer)

# char_layer = get_char_cnn(char_max_len,char_vocab_size,char_dim=30,name='char_Layer')
# Concatinating word from bidirectional LSTM and 1D CNN character output layer
# embed_layer = concatenate([embed_layer, char_layer], name='concat_layer')

# embed_layer = Dense(100, activation='sigmoid', name='Dense_concat_layer')(embed_layer)
# flat = Flatten()(embed_layer)
# final Dense layer for classes
cat_output = Dense(n_classes, activation='softmax', name='Final_Output_Layer')(embed_layer)

model = Model(inputs=w_tweet, outputs=cat_output, name='NER_Model')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# early_stopping = EarlyStopping(patience=20, verbose=1)
model.fit(X_train, y_train, epochs=10, batch_size=150,
          verbose=True, validation_data=[X_test, y_test], shuffle=True)

accuracy, loss_and_metrics = model.evaluate(X_test, y_test, verbose=0)
gc.collect()

predict = model.predict(X_test, verbose=True)

print(predict[0])

report = classification_report(K.argmax(y_test), predict)
print(report)

'''
pre_test_label_index = get_tag_index(p, 35, 21)
acc_test, f1_test,p_test,r_test=evaluate(pre_test_label_index, y_test,X_test,labelVoc,35,id_to_vocb)
pre_test_label_index_2 = pre_test_label_index.reshape(len(y_test)*35)
print('##test##, evaluate:''F1:',f1_test,'precision:',p_test,'recall:',r_test)
'''
