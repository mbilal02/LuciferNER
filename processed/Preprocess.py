from random import seed
from nltk.corpus.reader.conll import ConllCorpusReader
from nltk.tokenize.treebank import TreebankWordDetokenizer

from utilities.setting import DEV, TEST, TRAIN, TRAIN_M, DEV_M, TEST_M, TRAIN_, DEV_, TEST_
from utilities.utilities import create_lookup, create_sequences

seed(7)


def conllReader(corpus):
    '''
    Data reader for CoNLL format data
    '''
    root = "data/"
    sentences = []

    ccorpus = ConllCorpusReader(root, ".conll", ('words', 'pos', 'tree'))

    raw = ccorpus.sents(corpus)

    for sent in raw:
        sentences.append([TreebankWordDetokenizer().detokenize(sent)])

    tagged = ccorpus.tagged_sents(corpus)


    return tagged, sentences


def build_lookups(train_name, dev_name, test_name, voc):
    '''
    Wrapper function to builds lookups for characters and labels. Also, computes the sentence and word max lengths.
    '''
    sentences = []
    sentence = []
    sent_maxlen = 0
    word_maxlen = 0
    root = "data/"

    for fname in (train_name, dev_name, test_name):
        with open((root + fname), 'r', encoding='utf8') as file:
            for line in file:
                line = line.rstrip()
                if line == '':
                    sent_maxlen = max(sent_maxlen, len(sentence))
                    sentences.append(sentence)
                    sentence = []
                else:
                    sentence.append(line.split('\t'))
                    word_maxlen = max(word_maxlen, len(str(line.split()[0])))

    sentences.append(sentence)
    num_sentence = len(sentences)

    char_lookup, label_lookup = create_lookup(sentences, voc)

    # considring a higher valued max sent_length


    return [sentences, sent_maxlen, word_maxlen, num_sentence, char_lookup, label_lookup]


def build_image_sequences(train_name, dev_name, test_name, labels):
    """
    wrapper function to build image experiment related sequences
    """

    ID = 'IMGID'
    img_id = []
    sentences = []
    sentence = []
    sent_maxlen = 0
    word_maxlen = 0
    image_feature = []
    splits = []

    for data_file in (train_name, dev_name, test_name):
        splits.append(len(img_id))
        with open(('data/' + data_file), 'r') as file:
            for line in file:
                line = line.rstrip()
                if line == '':
                    sent_maxlen = max(sent_maxlen, len(sentence))
                    sentences.append(sentence)
                    sentence = []
                else:
                    if ID in line:
                        num = line[6:]
                        img_id.append(num)
                    else:
                        sentence.append(line.split('\t'))
                        word_maxlen = max(word_maxlen, len(str(line.split()[0])))

    sentences.append(sentence)
    splits.append(len(img_id))
    num_sentence = len(sentences)
    char_lookup, label_lookup = create_lookup(sentences, labels)
    '''
    for image in img_id:
        feature = img_feature_file.get(image)
        np_feature = np.array(feature)
        image_feature.append(np_feature)
    '''
    # considring a higher valued max sent_length
    X, Y, X_c, addChar = create_sequences(sentences, char_lookup, label_lookup,
                                          word_maxlen, sent_maxlen)
    return [splits, X, Y, X_c, addChar, sent_maxlen, word_maxlen, char_lookup, num_sentence]


def flatten(list):
    '''
    Helper function that flattens 2D lists.
    '''
    return [i for sublist in list for i in sublist]


def sequence_helper(x_in, sent_maxlen, casing=False):
    '''
    Helper function for word sequences (text data sepcific)
    :param x_in:
    :param sent_maxlen:
    :return: Word sequences
    '''

    new_X = []
    for seq in x_in:
        new_seq = []
        for i in range(sent_maxlen):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append('__pad__')
        new_X.append(new_seq)
    return new_X


def case_helper(x_in, sent_maxlen):
    '''
    Helper function for word sequences (text data sepcific)
    :param x_in:
    :param sent_maxlen:
    :return: Word sequences
    '''

    new_X = []
    for seq in x_in:
        new_seq = []
        for i in range(sent_maxlen):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append(int('7'))
        new_X.append(new_seq)
    return new_X


def start_build_sequences(vocabulary):
    '''
    Sequence builder for text data specific
    :param vocabulary: Label vocabulary specific for the datasets
    :return: Sequences
    '''
    sentences, sent_maxlen, word_maxlen, \
    num_sentence, char_lookup, label_lookup = build_lookups(TRAIN_, DEV_, TEST_, vocabulary)
    train_sent, train_dt_sent = conllReader(TRAIN_)
    dev_sent, dev_dt_sent = conllReader(DEV_)
    test_sent, test_dt_sent = conllReader(TEST_)



    # logger.info('Setting up input sequences')
    x, y, x_c, addCharTrain = create_sequences(train_sent, char_lookup, label_lookup, word_maxlen, sent_maxlen)
    x_t, y_t, xc_t, addCharTest = create_sequences(test_sent, char_lookup, label_lookup, word_maxlen, sent_maxlen)
    x_d, y_d, xc_d, addCharDev = create_sequences(dev_sent, char_lookup, label_lookup, word_maxlen, sent_maxlen)
    X_train = sequence_helper(x, sent_maxlen)
    X_test = sequence_helper(x_t, sent_maxlen)
    X_dev = sequence_helper(x_d, sent_maxlen)
    caseTrain = case_helper(addCharTrain, sent_maxlen)
    caseTest = case_helper(addCharTest, sent_maxlen)
    caseDev = case_helper(addCharDev, sent_maxlen)
    print(caseTrain)

    return [train_dt_sent, dev_dt_sent, test_dt_sent, X_train, X_dev, X_test, x_c, xc_d, xc_t, y, y_d, y_t, caseTrain,
            caseDev,
            caseTest, char_lookup,
            sent_maxlen, word_maxlen]


def start_build_image_sequences(vocabulary):
    '''
    Builds train, dev, and test sequences ready for training and testing in multimidal setting
    '''
    splits, X, Y, X_c, X_img, sent_maxlen, word_maxlen, char_lookup, num_sentence = build_image_sequences(TRAIN_M,
                                                                                                          DEV_M, TEST_M,
                                                                                                          vocabulary)
    num_train = splits[1]
    num_dev = splits[2] - splits[1]

    train_x = X[:num_train]
    train_x_c = X_c[:num_train]
    train_y = Y[:num_train]
    train_img_x = X_img[:num_train]

    dev_x = X[num_train:num_train + num_dev]
    dev_x_c = X_c[num_train:num_train + num_dev]
    dev_y = Y[num_train:num_train + num_dev]
    dev_img_x = X_img[num_train:num_train + num_dev]

    test_x = X[num_train + num_dev:]
    test_x_c = X_c[num_train + num_dev:]
    test_y = Y[num_train + num_dev:]
    test_img_x = X_img[num_train + num_dev:]

    X_train = sequence_helper(train_x, sent_maxlen)
    X_test = sequence_helper(test_x, sent_maxlen)
    X_dev = sequence_helper(dev_x, sent_maxlen)

    return [X_train, X_test, X_dev, train_x_c, dev_x_c, test_x_c,
            train_y, dev_y, test_y, train_img_x, dev_img_x, test_img_x, sent_maxlen, word_maxlen, char_lookup]
