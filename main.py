import logging

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
from numpy.random import seed

from algorithm.network import network_model, EXTENDED_SENTENCE_MODEL
from evaluation.eval import Evaluator
from evaluation.eval_script import get_wnut_evaluation
from processed.Preprocess import start_build_sequences
from utilities.setting import BASE_MODEL, EXTENDED_BASE_MODEL, SIMPLE_TEXT_ATTENTION, SEGREGATED_TEXT_ATTENTION, D, \
    conll03
from utilities.utilities import getLabels, save_predictions



seed(1200) #7 #1337 #879

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LuciferNER:
    def __init__(self, architecture, batch_size, n_epochs, patience, lr_r):
        self.architecture = architecture
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.patience = patience
        self.lr_r = lr_r

    def run(self, filename=None, dataset_type=None, model_file=None, label_vocab=None, label_key=None):
        '''
        Builds an NER model, predicts, saves prediction files, loads evaulation
        F1-scores from eval script (WNUT 2017 evaluation script cited in the paper)
        '''

        logger.info('Preparing data initiated')
        train_sent, dev_sent, test_sent, X_train, X_dev, X_test, x_c, xc_d, xc_t, y, y_d, \
        y_t, addCharTrain, addCharDev, \
        addCharTest, char_lookup, sent_maxlen, word_maxlen = start_build_sequences(
            vocabulary=conll03)
        print(sent_maxlen)
        y = y.reshape(y.shape[0], y.shape[1], 1)
        y_t = y_t.reshape(y_t.shape[0], y_t.shape[1], 1)
        y_d = y_d.reshape(y_d.shape[0], y_d.shape[1], 1)

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
        if self.architecture == BASE_MODEL:
            model.fit([np.array(x_c), np.array(X_train)], y,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      verbose=1,
                      callbacks=[checkpointer, earlystopper],
                      validation_data=([np.array(xc_d), np.array(X_dev)], y_d),
                      shuffle=True)
            predict = model.predict([np.array(xc_t), np.array(X_test)],
                                    verbose=1,
                                    batch_size=self.batch_size)
            self.get_prediction(X_test,
                                           y_t,
                                           predict,
                                           filename,
                                           label_vocab,
                                           dataset_type,
                                           label_key)
            #get_wnut_evaluation(filename)

        elif self.architecture == EXTENDED_SENTENCE_MODEL:
            #with tf.device('/device:GPU:0'):
                model.fit([np.array(x_c),np.array(X_train),np.array(addCharTrain), np.array(train_sent)], y,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          verbose=1,
                          callbacks=[checkpointer, earlystopper],
                          validation_data=([np.array(xc_t),np.array(X_test),np.array(addCharTest), np.array(test_sent)], y_t),
                          shuffle=True)

                predict = model.predict([np.array(xc_t),np.array(X_test),np.array(addCharTest), np.array(test_sent)],
                                        verbose=1,
                                        batch_size=self.batch_size)
                self.get_prediction(X_test, y_t, predict, filename, label_vocab, dataset_type,
                                               label_key)
            #get_wnut_evaluation(filename)


        else:

                model.fit([np.array(x_c), np.array(X_train), np.array(addCharTrain)], y,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          verbose=1,
                          callbacks=[checkpointer, earlystopper],
                          validation_data=([np.array(xc_d), np.array(X_dev), np.array(addCharDev)], y_d),
                          shuffle=True)
                predict = model.predict([np.array(xc_t), np.array(X_test), np.array(addCharTest)],
                                        verbose=1,
                                        batch_size=self.batch_size)
                self.get_prediction(X_test, y_t, predict, filename, label_vocab, dataset_type
                                               ,label_key)
                #get_wnut_evaluation(filename)



    def get_prediction(self, x, y, predict, filename, label_vocab, label_key, dataset_type):

        prediction = np.argmax(predict, axis=-1)
        prediction_final = np.array(prediction).tolist()
        predictions = getLabels(prediction_final, vocabulary=label_vocab)
        true = getLabels(y, vocabulary=label_vocab)
        save_predictions(filename, x, true, predictions)
        get_wnut_evaluation(filename)
        evaluator = Evaluator(true=true, pred=predictions, tags=label_key)
        res, agg= evaluator.evaluate()
        print(res)
        print(agg)


if __name__ == '__main__':
    label_a = ['tv-show', 'person', 'product', 'music-artist', 'movie', 'facility', 'company', 'geo-loc', 'other',
               'sportsteam']
    label_b = ['person', 'location', 'creative-work', 'corporation', 'product', 'group']
    label_m = ['PER', 'ORG', 'LOC', 'OTHER']
    label_c = ['PER', 'ORG', 'LOC', 'MISC']
    experiments = [BASE_MODEL, EXTENDED_BASE_MODEL, EXTENDED_SENTENCE_MODEL, SIMPLE_TEXT_ATTENTION, SEGREGATED_TEXT_ATTENTION]


    ner = LuciferNER(architecture=EXTENDED_SENTENCE_MODEL,
                         batch_size=500,
                         n_epochs=100,
                         patience=3,
                         lr_r=0.001)
    ner.run(filename='sent_att.tsv',
                                       dataset_type=D,
                                       model_file='textual_model21',
                                       label_vocab=conll03,
            label_key=label_c)

    print('/------------------------End Experiment------------------------------/')
