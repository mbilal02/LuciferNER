import gc
import os
import pickle

import talos

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, RMSprop, Nadam
from sklearn.metrics import f1_score, classification_report
from talos.model import lr_normalizer

from algorithm.network import simple_word_level_model
from evaluation.eval_script import get_wnut_evaluation
from processed.Preprocess import start_build_sequences, flatten
import numpy as np
import logging
from numpy.random import seed

from utilities.setting import B, wnut_b, multi_modal, C
from utilities.utilities import getLabels, save_predictions
import time

seed(7)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sml = 0


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, protocol=2)


def project_object(obj, *attributes):
    out = {}
    for a in attributes:
        out[a] = getattr(obj, a)
    return out


def NERmodel(x_train, y_train, x_val, y_val, params):
    global sml
    model = simple_word_level_model(sml, B, params)
    rms = RMSprop(lr=params['lr'], rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['acc', talos.utils.metrics.f1score])
    model.summary()
    #checkpointer = ModelCheckpoint(filepath='models/w_c_best_model2019.hdf5', verbose=1, save_best_only=True)
    #earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    out = model.fit(x_train, y_train,
                    epochs=50, batch_size=params['batch'], verbose=1,
                    validation_data=(x_val, y_val), shuffle=True)

    return out, model


def start_experiment():
    '''
    Builds an NER model, predicts, saves prediction files, loads evaulation
    F1-scores from eval script (WNUT 2017 evaluation script cited in the paper)
    '''
    parameters = {
        'units': [1024, 512, 300, 200, 100, 64],
        'dropout': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.25, 0.15],
        'hidden_units': [1024, 512, 300, 200, 100, 64],
        'lr': [0.19,0.01, 0.001, 0.0001],
        'optimizer':['nadam', 'adam', 'rmsprop', 'sgd'],
        'batch':[100]
    }

    print('Preparing data sequences ')
    X_train, X_dev, X_test, y, y_t, y_d, sent_maxlen, word_maxlen = start_build_sequences(
        vocabulary=wnut_b)

    global sml
    sml = sent_maxlen

    print('Now setting up a basic NER model')
    y_train = y.reshape(y.shape[0], y.shape[1], 1)
    y_val = y_d.reshape(y_d.shape[0], y_d.shape[1], 1)
    x_train = [np.array(X_train)]
    x_val = [np.array(X_dev)]

    print('Training initiated. This may take long time to conclude..')
    print(
        'WARNING: Please check if you are using GPU to train your model, else, use Keras Backend to enable GPU usage.')

    tt = talos.Scan(x=x_train
               , y=y_train
               , params=parameters
               , model=NERmodel
               , x_val=x_val
               , y_val=y_val
               , experiment_name='first'
               , round_limit=10
               )
    t = project_object(tt, 'params', 'saved_models', 'saved_weights', 'data', 'details', 'round_history')
    save_object(t, 'result.pickle')
    gc.collect()

    # model.save('Ner_word_character.h5')
    # print('Model testing step initiated...')
    # y_t = y_t.reshape(y_t.shape[0], y_t.shape[1], 1)
    # predict = model.predict([np.array(X_test)], verbose=1, batch_size=50)
    # prediction = np.argmax(predict, axis=-1)
    #
    # truth = y_t
    # print('Building evaluation results')
    # prediction_final = np.array(prediction).tolist()
    #
    # predictionss = getLabels(prediction_final, vocabulary=wnut_b)
    # print('True:'+str(y_t[5]))
    # print('Predicted'+str(prediction[5]))
    # print('True:'+str(y_t[6]))
    # print('Predicted'+str(prediction[6]))
    # print('-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/')
    # print('Aroused your excitement?')
    #
    # true = getLabels(truth, vocabulary=wnut_b)
    # save_predictions('t.tsv', X_test, true, predictionss)


if __name__ == '__main__':
    start_time = time.time()
    start_experiment()
    print("\nThis took %s minutes to run !!" % ((time.time() - start_time) / 60))
