import gc
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, RMSprop, Nadam
from sklearn.metrics import f1_score, classification_report
from algorithm.network import build_bilstm_cnn_model, build_multimodal_attnetion_network
from evaluation.eval_script import get_wnut_evaluation
from processed.Preprocess import start_build_sequences, flatten, start_build_image_sequences
import numpy as np
import logging
from numpy.random import seed

from utilities.setting import B, wnut_b, multi_modal, C
from utilities.utilities import getLabels, save_predictions

seed(7)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def start_NER_Model():
    '''
    Builds an NER model, predicts, saves prediction files, loads evaulation
    F1-scores from eval script (WNUT 2017 evaluation script cited in the paper)
    '''
    logger.info('Preparing data initiated')
    X_train, X_dev, X_test, x_c, xc_d, xc_t, y, y_d, y_t, addCharTrain, addCharDev,\
    addCharTest, char_lookup, sent_maxlen, word_maxlen = start_build_sequences(
        vocabulary=wnut_b)
    print(len(X_train))
    print(len(X_test))
    print(len(X_dev))
    print(X_train)

    model = build_bilstm_cnn_model(sent_maxlen, word_maxlen, char_lookup, dataset_type=B)

    # nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    y = y.reshape(y.shape[0], y.shape[1], 1)
    y_t = y_t.reshape(y_t.shape[0], y_t.shape[1], 1)
    y_d = y_d.reshape(y_d.shape[0], y_d.shape[1], 1)
    checkpointer = ModelCheckpoint(filepath='models/w_c_best_model2019.hdf5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    logger.info('Training initiated. This may take long time to conclude..')
    model.fit([np.array(X_train), np.array(x_c), np.array(addCharTrain)], y,
              epochs=12, batch_size=50, verbose=1, callbacks=[checkpointer, earlystopper],
              validation_data=([np.array(X_dev), np.array(xc_d), np.array(addCharDev)], y_d), shuffle=True)
    gc.collect()

    # model.save('Ner_word_character.h5')
    logger.info('Model testing step initiated...')
    predict = model.predict([np.array(X_test), np.array(xc_t), np.array(addCharTest)], verbose=1, batch_size=50)
    prediction = np.argmax(predict, axis=-1)
    #print(prediction[4])

    truth = y_t
    logger.info('Building evaluation results')
    #print(classification_report(np.array(truth), np.array(flatten(prediction))))
    prediction_final = np.array(prediction).tolist()

    predictionss = getLabels(prediction_final, vocabulary=wnut_b)
    print(predictionss[4])

    #truth = getLabels(np.array(y_t).tolist(), vocabulary=wnut_b)

    #save_predictions('exp1a.tsv', flatten(X_test), truth, predictionss)
    true = getLabels(truth, vocabulary=wnut_b)
    save_predictions('exp1b.tsv', flatten(X_test), true, predictionss)
    #get_wnut_evaluation()


def start_multimodal_ner_model():
    """
    Builds a multi-modal NER model, predicts, saves prediction files, loads evaulation
    F1-scores from eval script (WNUT 2017 evaluation script cited in the paper)
    """

    logger.info('Preparing data initiated...')
    X_train, X_test, X_dev, train_x_c, dev_x_c, test_x_c, \
    train_y, dev_y, test_y, train_img_x, dev_img_x, test_img_x, \
    sent_maxlen, word_maxlen, char_lookup = start_build_image_sequences(vocabulary=multi_modal)

    model = build_multimodal_attnetion_network(sent_maxlen, word_maxlen, char_lookup, dataset_type=C)

    # nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    train_y = train_y.reshape(train_y.shape[0], train_y.shape[1], 1)
    test_y = test_y.reshape(test_y.shape[0], test_y.shape[1], 1)
    dev_y = dev_y.reshape(dev_y.shape[0], dev_y.shape[1], 1)
    checkpointer = ModelCheckpoint(filepath='models/w_c_v_best_model.hdf5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    logger.info('Training initiated. This may take long time to conclude..')
    model.fit([np.array(X_train), np.array(train_x_c)], train_y,
              epochs=18, batch_size=50, verbose=1, callbacks=[checkpointer, earlystopper],
              validation_data=([np.array(X_dev), np.array(dev_x_c)], dev_y), shuffle=True)
    gc.collect()

    # model.save('Ner_word_character.h5')
    logger.info('Prediction step initiating...')

    predict = model.predict([np.array(X_test), np.array(test_x_c)], verbose=1, batch_size=20)
    prediction = np.argmax(predict, axis=-1)

    truth = flatten(test_y)
    logger.info('Building evaluation results')
    print(classification_report(np.array(truth), np.array(flatten(prediction))))
    prediction_final = np.array(prediction).tolist()
    predictionss = getLabels(prediction_final, vocabulary=multi_modal)

    truth = getLabels(np.array(test_y).tolist(), vocabulary=multi_modal)

    save_predictions('evaluation/submission.tsv', X_test, truth, predictionss)

    get_wnut_evaluation()


if __name__ == '__main__':
    start_NER_Model()
