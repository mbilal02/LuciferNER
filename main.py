import gc
import logging

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
from numpy.random import seed

from algorithm.network import build_bilstm_cnn_model, network_model
from evaluation.eval import Evaluator
from evaluation.eval_script import get_wnut_evaluation
from processed.Preprocess import start_build_sequences
from utilities.setting import B, wnut_b, BASE_MODEL, EXTENDED_BASE_MODEL, SIMPLE_TEXT_ATTENTION, wnut_a, \
    SEGREGATED_TEXT_ATTENTION
from utilities.utilities import getLabels, save_predictions
import tensorflow as tf
from keras import backend as K
NUM_CORES= 16
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_CORES,
                        inter_op_parallelism_threads=NUM_CORES,
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 1}
                       )
session = tf.Session(config=config)
K.set_session(session)

seed(7)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

label_a = ['tv-show', 'person', 'product', 'music-artist', 'movie', 'facility', 'company', 'geo-loc', 'other',
           'sportsteam']
label_b = ['person', 'location', 'creative-work', 'corporation', 'product', 'group']
label_m = ['PER', 'ORG', 'LOC', 'OTHER', 'product']

class LuciferNER:

    def __init__(self, architecture,  batch_size, n_epochs, patience, lr_r):
        self.architecture = architecture
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.patience = patience
        self.lr_r = lr_r

    def run(self, filename=None, dataset_type=None, model_file=None, label_vocab=None):
            '''
            Builds an NER model, predicts, saves prediction files, loads evaulation
            F1-scores from eval script (WNUT 2017 evaluation script cited in the paper)
            '''

            logger.info('Preparing data initiated')
            train_sent, dev_sent, test_sent, X_train, X_dev, X_test, x_c, xc_d, xc_t, y, y_d,\
            y_t, addCharTrain, addCharDev, \
            addCharTest, char_lookup, sent_maxlen, word_maxlen = start_build_sequences(
                vocabulary=wnut_b)
            y = y.reshape(y.shape[0], y.shape[1], 1)
            y_t = y_t.reshape(y_t.shape[0], y_t.shape[1], 1)
            y_d = y_d.reshape(y_d.shape[0], y_d.shape[1], 1)

            if self.architecture == BASE_MODEL:
                model = network_model(sent_maxlen,
                                      word_maxlen,
                                      char_lookup,
                                      dataset_type,
                                      architecture=self.architecture)
                checkpointer = ModelCheckpoint(filepath='models/' + model_file + '.hdf5',
                                               verbose=1,
                                               save_best_only=True)
                earlystopper = EarlyStopping(monitor='val_loss',
                                             patience=self.patience,
                                             verbose=1)
                rms = RMSprop(lr=self.lr_r,
                              rho=0.9,
                              epsilon=None,
                              decay=0.0)
                model.compile(optimizer=rms,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                model.summary()
                model.fit([np.array(x_c), np.array(X_train)], y,
                          epochs=self.epochs, batch_size=self.batch_size, verbose=1,
                          callbacks=[checkpointer, earlystopper],
                          validation_data=([np.array(xc_d), np.array(X_dev)], y_d), shuffle=True)
                predict = model.predict([np.array(xc_t), np.array(X_test)], verbose=1, batch_size=self.batch_size)
                res, agg = self.get_prediction(X_test, y_t, predict, filename, label_vocab, dataset_type)
                get_wnut_evaluation(filename)
                return res, agg

            elif self.architecture:
                model = network_model(sent_maxlen, word_maxlen, char_lookup, dataset_type,
                                      architecture=self.architecture)
                checkpointer = ModelCheckpoint(filepath='models/' + model_file + '.hdf5',
                                               verbose=1,
                                               save_best_only=True)

                earlystopper = EarlyStopping(monitor='val_loss',
                                             patience=self.patience,
                                             verbose=1)

                rms = RMSprop(lr=self.lr_r,
                              rho=0.9,
                              epsilon=None,
                              decay=0.0)
                model.compile(optimizer=rms,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                model.summary()
                model.fit([np.array(x_c), np.array(X_train), np.array(addCharTrain)], y,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          verbose=1,
                          callbacks=[checkpointer, earlystopper],
                          validation_data=([np.array(xc_d), np.array(X_dev), np.array(addCharDev)],y_d),
                          shuffle=True)
                predict = model.predict([np.array(xc_t),
                                         np.array(X_test),
                                         np.array(addCharTest)],
                                        verbose=1, batch_size=self.batch_size)
                res, agg = self.get_prediction(X_test, y_t, predict, filename, label_vocab, dataset_type)
                get_wnut_evaluation(filename)
                return res, agg

            elif self.architecture == SIMPLE_TEXT_ATTENTION:
                model = network_model(sent_maxlen, word_maxlen, char_lookup, dataset_type,
                                      architecture=self.architecture)
                checkpointer = ModelCheckpoint(filepath='models/' + model_file + '.hdf5',
                                               verbose=1,
                                               save_best_only=True)
                earlystopper = EarlyStopping(monitor='val_loss',
                                             patience=self.patience,
                                             verbose=1)
                rms = RMSprop(lr=self.lr_r, rho=0.9, epsilon=None, decay=0.0)
                model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.summary()
                model.fit([np.array(x_c), np.array(X_train), np.array(addCharTrain)], y,
                          epochs=self.epochs,
                          batch_size=self.batch_size, verbose=1,
                          callbacks=[checkpointer, earlystopper],
                          validation_data=([np.array(xc_d), np.array(X_dev), np.array(addCharDev)], y_d),
                          shuffle=True)
                predict = model.predict([np.array(xc_t),
                                         np.array(X_test),
                                         np.array(addCharDev)],
                                         verbose=1,
                                         batch_size=self.batch_size)
                res, agg = self.get_prediction(X_test,
                                               y_t,
                                               predict,
                                               filename,
                                               label_vocab,
                                               dataset_type)
                get_wnut_evaluation(filename)
                return res, agg

            elif self.architecture == SEGREGATED_TEXT_ATTENTION:
                model = network_model(sent_maxlen,
                                      word_maxlen,
                                      char_lookup,
                                      dataset_type,
                                      architecture=self.architecture)
                checkpointer = ModelCheckpoint(filepath='models/' + model_file + '.hdf5',
                                               verbose=1,
                                               save_best_only=True)
                earlystopper = EarlyStopping(monitor='val_loss',
                                             patience=self.patience,
                                             verbose=1)
                rms = RMSprop(lr=self.lr_r,
                              rho=0.9,
                              epsilon=None,
                              decay=0.0)

                model.compile(optimizer=rms,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                model.summary()
                model.fit([np.array(x_c), np.array(X_train), np.array(addCharTrain)], y,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          verbose=1,
                          callbacks=[checkpointer, earlystopper],
                          validation_data=([np.array(xc_d), np.array(X_dev), np.array(addCharDev)], y_d),
                          shuffle=True)
                predict = model.predict([np.array(xc_t),
                                         np.array(X_test),
                                         np.array(addCharDev)],
                                         verbose=1,
                                         batch_size=self.batch_size)
                res, agg = self.get_prediction(X_test,
                                               y_t,
                                               predict,
                                               filename,
                                               label_vocab,
                                               dataset_type)
                get_wnut_evaluation(filename)
                return res, agg




    def get_prediction(self,x, y, predict, filename, label_vocab, dataset_type):
        prediction = np.argmax(predict, axis=-1)
        prediction_final = np.array(prediction).tolist()
        predictions = getLabels(prediction_final, vocabulary=label_vocab)
        true = getLabels(y, vocabulary=label_vocab)
        save_predictions(filename, x, true, predictions)
        get_wnut_evaluation(filename)
        if dataset_type == wnut_b:
            evaluator = Evaluator(true=true, pred=predictions, tags=label_b)
            return evaluator.evaluate()
        elif dataset_type == wnut_a:
            evaluator = Evaluator(true=true, pred=predictions, tags=label_a)
            return evaluator.evaluate()
        else:
            evaluator = Evaluator(true=true, pred=predictions, tags=label_m)
            return evaluator.evaluate()


if __name__ == '__main__':
    experiments = [EXTENDED_BASE_MODEL,SIMPLE_TEXT_ATTENTION, SEGREGATED_TEXT_ATTENTION]

    for id, exp in enumerate(experiments, start= 0):
        ner = LuciferNER(architecture=exp,
                         batch_size= 100,
                         n_epochs= 100,
                         patience= 10,
                         lr_r= 0.001)
        results, results_agg = ner.run( filename='00'+str(id)+'.tsv',
                                        dataset_type=B,
                                        model_file='textual'+str(id)+'model',
                                        label_vocab=wnut_b)

        print('Building results for'.format(exp))
        print(results)
        print(results_agg)
        print('/------------------------End Experiment------------------------------/')
