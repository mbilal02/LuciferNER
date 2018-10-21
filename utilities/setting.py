from enum import Enum

import h5py


######################################################
#                     Important constants            #
######################################################
A = 23
B = 13
C = 9
UNK_TOKEN= '<UNK>'
PAD_TOKEN= '<PAD>'

######################################################
#                     Data Set Paths                 #
######################################################
# WNUT 2017

TRAIN = "training.conll"

DEV = "dev.conll"

TEST = "test.conll"


#WNUT 2016

TRAIN_ = "train_2016"
DEV_ = "dev_2016"
TEST_ = "test_2016"

#Multimodal dataset

TRAIN_M = "train_i"
DEV_M = "dev_i"
TEST_M = "test_i"
feat_file = 'models/feat_file.h5'
img_feature_file = h5py.File((feat_file), 'r')



######################################################
#                     Label vocabularies             #
######################################################


wnut_a = {
        'I-tvshow': 20,
        'B-tvshow': 19,
        'I-sportsteam': 18,
        'B-sportsteam': 17,
        'I-product': 16,
        'B-product': 15,
        'I-person': 14,
        'B-person': 13,
        'I-other': 12,
        'B-other': 11,
        'I-musicartist': 10,
        'B-musicartist': 9,
        'I-movie': 8,
        'B-movie': 7,
        'I-geo-loc': 6,
        'B-geo-loc': 5,
        'I-facility': 4,
        'B-facility': 3,
        'I-company': 2,
        'B-company': 1,
        'O': 0}


wnut_b = { 'B-corporation':12,
    'B-creative-work':11,
    'B-group':10,
    'B-location':9,
    'B-person':8,
    'B-product':7,
    'I-corporation':6,
    'I-creative-work':5,
    'I-group':4,
    'I-location':3,
    'I-person':2,
    'I-product':1,
    'O':0}

multi_modal= {'O':0,
	'B-PER':1, 'I-PER':2,
	'B-LOC':3, 'I-LOC':4,
	'B-ORG':5, 'I-ORG':6,
	'B-OTHER':7, 'I-OTHER':8,
	}

