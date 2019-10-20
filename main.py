import gc
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, RMSprop, Nadam
from sklearn.metrics import f1_score, classification_report

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
def start_NER_Model():
    '''
    Builds an NER model, predicts, saves prediction files, loads evaulation
    F1-scores from eval script (WNUT 2017 evaluation script cited in the paper)
    '''
    print('Preparing data sequences ')
    X_train, X_dev, X_test, y, y_t, y_d, sent_maxlen, word_maxlen = start_build_sequences(
        vocabulary=wnut_b)
    print(str(len(X_train)) + ','+  str(len(X_dev)) + ',' + str(len(X_test)) ) #that means the batch size could be a divisible number
    print('Now setting up a basic NER model')
    model = simple_word_level_model(sent_maxlen, dataset_type=B)

    # nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    y = y.reshape(y.shape[0], y.shape[1], 1)
    y_t = y_t.reshape(y_t.shape[0], y_t.shape[1], 1)
    y_d = y_d.reshape(y_d.shape[0], y_d.shape[1], 1)
    checkpointer = ModelCheckpoint(filepath='models/w_c_best_model2019.hdf5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    print('Training initiated. This may take long time to conclude..')
    print('WARNING: Please check if you are using GPU to train your model, else, use Keras Backend to enable GPU usage.')
    model.fit([np.array(X_train)], y,
              epochs=12, batch_size=50, verbose=1, callbacks=[checkpointer, earlystopper],
              validation_data=([np.array(X_dev)], y_d), shuffle=True)
    gc.collect()

    # model.save('Ner_word_character.h5')
    print('Model testing step initiated...')
    predict = model.predict([np.array(X_test)], verbose=1, batch_size=50)
    prediction = np.argmax(predict, axis=-1)
    #print(prediction[4])

    truth = y_t
    print('Building evaluation results')
    #print(classification_report(np.array(truth), np.array(flatten(prediction))))
    prediction_final = np.array(prediction).tolist()

    predictionss = getLabels(prediction_final, vocabulary=wnut_b)
    print('True:'+str(y_t[5]))
    print('Predicted'+str(prediction[5]))
    print('True:'+str(y_t[6]))
    print('Predicted'+str(prediction[6]))
    print('-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/')
    print('Aroused your excitement?')

    # truth = getLabels(np.array(y_t).tolist(), vocabulary=wnut_b)

    # save_predictions('exp1a.tsv', flatten(X_test), truth, predictionss)
    true = getLabels(truth, vocabulary=wnut_b)
    save_predictions('t.tsv', X_test, true, predictionss)




if __name__ == '__main__':
    start_time = time.time()
    start_NER_Model()
    print("\nThis took %s minutes to run !!" % ((time.time() - start_time)/60))
