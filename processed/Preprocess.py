from random import seed
from nltk.corpus.reader.conll import ConllCorpusReader
from utilities.setting import DEV, TEST, TRAIN
from utilities.utilities import create_lookup, create_sequences

seed(7)


def conllReader(corpus):
    '''
    Data reader for CoNLL format data
    '''
    root = "data/"

    ccorpus = ConllCorpusReader(root, ".conll", ('words', 'pos', 'tree'))

    return ccorpus.tagged_sents(corpus)


def getSentenceInfo(train_name, dev_name, test_name, voc):
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
    #num_sentence = len(sentences)

    # At the moment you do need to care about Character look ups, only label look up will be fine.
    label_lookup = create_lookup(sentences, voc)

    # considring a higher valued max sent_length

    return [sentences, sent_maxlen, word_maxlen, label_lookup]


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


def start_build_sequences(vocabulary):
    '''
    Sequence builder for text data specific
    :param vocabulary: Label vocabulary specific for the datasets
    :return: Sequences
    '''
    sentences, sent_maxlen, word_maxlen, label_lookup = getSentenceInfo(TRAIN, DEV, TEST, vocabulary)
    train_sent = conllReader(TRAIN)
    dev_sent = conllReader(DEV)
    test_sent = conllReader(TEST)
    # logger.info('Setting up input sequences')
    x, y, sent_maxlen, word_maxlen = create_sequences(train_sent,label_lookup, word_maxlen, sent_maxlen)
    x_t, y_t,sent_maxlen, word_maxlen = create_sequences(test_sent, label_lookup, word_maxlen, sent_maxlen)
    x_d, y_d,sent_maxlen, word_maxlen = create_sequences(dev_sent, label_lookup, word_maxlen, sent_maxlen)
    #pad to sentence maxlength because we have sentences that are nt of the same length
    X_train = sequence_helper(x, sent_maxlen)
    X_test = sequence_helper(x_t, sent_maxlen)
    X_dev = sequence_helper(x_d, sent_maxlen)

    return [X_train, X_dev, X_test,y, y_t,y_d, sent_maxlen, word_maxlen]
