import NNAlign
import numpy as np
import encoding
import tempfile
import pickle
import math
import itertools
import sys
import random
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
def pad_encode_peptide(peptide, pad_dict, aa_encoder, wildcard_encoding):
    encoded = []
    peptide_index = 0
    pad = pad_dict[len(peptide)]
    for i in range(0, len(pad)):
        assert(pad[i] == 'A' or pad[i] == '-')
        if pad[i] == 'A':
            encoded.append(aa_encoder.encode_aa(peptide[peptide_index]))
            peptide_index += 1
        else:
            encoded.append(wildcard_encoding)
    return np.array(encoded)

def encode_data(data, wildcard_encoding):
    aa_encoder = encoding.BlossumEncoder()
    pad_dict = None
    train_x = []
    with open('padding.pickle', 'rb') as f:
        pad_dict = pickle.load(f)
    for peptide,pocket,measure in data:
        encoded_peptide = pad_encode_peptide(peptide, pad_dict, aa_encoder, wildcard_encoding)
        train_x.append(NNAlign.core_encoding(encoded_peptide, len(peptide), pocket, 9))
    return np.array(train_x)

"""
train_set should be a list of tuples of the form (peptide_encoding, pocket_encoding, log-transformed ic50)
"""
def train_padded(model, train_set, wildcard_encoding, weights_location):
    train_list = []
    train_x = []
    train_y = []
    aa_encoder = encoding.BlossumEncoder()
    pad_dict = None
    with open('padding.pickle', 'rb') as f:
        pad_dict = pickle.load(f)
    for peptide,pocket,measure in train_set:
        encoded_peptide = pad_encode_peptide(peptide, pad_dict, aa_encoder, wildcard_encoding)
        train_x.append(NNAlign.core_encoding(encoded_peptide, len(peptide), pocket, 9))
        train_y.append(measure)
    
    model.fit(np.array(train_x), np.array(train_y), epochs=1000
    plot_model(model, to_file='padded.png')
    model.summary()
    model.save_weights(weights_location)
    return model
