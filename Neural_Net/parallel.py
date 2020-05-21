import random
from keras import optimizers
import math
import NNPad
import sys
import encoding
from keras.models import Sequential
from keras.layers import Dense, Activation
from multiprocessing import Pool
import numpy as np
import csv
import pickle
import NNAlign
from collections import defaultdict
import cnn_copy
from sklearn.model_selection import KFold
from scipy import stats
def undo_log(x):
    return math.e**((1 - x)*math.log(50000))

pg = encoding.PositionGetter('pocket_positions.pickle')
positions = pg.get_pocket_positions('A'*9)

pocket_dict = pg.get_dict()
pocket_positions = list()
for v in pocket_dict.values():
    pocket_positions.extend(v)


length_to_pocket_to_peptide = None
with open('length_to_pocket_to_peptide.pickle', 'rb') as f:
    length_to_pocket_to_peptide = pickle.load(f)
unique_pocket_positions = list(set(pocket_positions))

pseudosequences = encoding.get_pseudosequences_dict('mhc_mapper.csv', 'complete_mhc.fasta', unique_pocket_positions)
pseudosequences_align = encoding.get_pseudosequences_dict('mhc_mapper.csv', 'complete_mhc.fasta', positions)
blencoder = encoding.BlossumEncoder()
def encode_sequence(sequence):
    """
    letters = ['A', 'C', 'D', 'E', 'F', 'G' ,'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    a = [0]*len(letters)
    encoded = []
    for x in sequence:
        a = [0.1]*len(letters)
        a[letters.index(x)] = 0.9
        encoded.append(np.array(a))
    return np.array(encoded)
    """
    return np.array([blencoder.encode_aa(x) for x in sequence])


encoded_pseudosequences_align = {k: encode_sequence(v) for k,v in pseudosequences_align.items()}
encoded_pseudosequences = {k: encode_sequence(v) for k,v in pseudosequences.items()}
cnn_data = []
padded_data = []
nnalign_data = []
alleles = []
measures = []
random.seed(1)
with open('ic50.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for x in reader:
        peptide = x['Peptide']
        encoded_peptide = encode_sequence(peptide)
        measure = float(x['1-log50k'])
        allele = x['Allele']
        if allele in encoded_pseudosequences and len(peptide) <= 13:
            encoded_allele_align = encoded_pseudosequences_align[allele]
            encoded_allele = encoded_pseudosequences[allele]
            partition = int(x['Partition'])
            alleles.append(allele)
            padded_data.append((peptide, encoded_allele_align, measure))
            nnalign_data.append((encoded_peptide, encoded_allele_align, measure))
            cnn_data.append((peptide, pseudosequences[allele], measure))
            measures.append(measure)

assert len(alleles) == len(padded_data)
assert len(padded_data) == len(nnalign_data)
assert len(cnn_data) == len(nnalign_data)
wildcard_encoding = np.array([0.0]*len(encode_sequence('A')[0]))
input_dim = 4 + (9 + len(positions))*wildcard_encoding.shape[0]

def train_and_performance(data):
    print('data')
    train_indices = data[0]
    thing = [cnn_data[i] for i in train_indices]
    pocket_lengths = set()
    for x in thing:
        pocket_lengths.add(len(x[1]))
    test_indices = data[1]
    model_type = data[2]
    weights_location = data[3]
    print('weights location')
    print(weights_location)
    print('test indices: %s' % str(test_indices))
    print(test_indices)
    """
    model_type is either 'cnn', 'align', or 'padded'

    Returns: (pearson, top_fraction)


    """
    assert model_type in ['cnn', 'align', 'padded']
    predictions = None
    if model_type == 'cnn':
        model = cnn_copy.train_cnn(56, [cnn_data[i] for i in train_indices], unique_pocket_positions, length_to_pocket_to_peptide, weights_location)
        test_data = cnn_copy.encode_data([cnn_data[i] for i in test_indices])
        print('test data')
        print(test_data)
        if test_data.size < 5:
            return (0, 0)
        predictions = list(model.predict(test_data).flatten())
    elif model_type == 'align':
        model = Sequential([
            Dense(56, input_dim=input_dim, activation='sigmoid'),
            Dense(1, activation='sigmoid')])
        model.compile('RMSProp', loss='mean_absolute_error')
        NNAlign.train_nnalign(model, [nnalign_data[i] for i in train_indices], wildcard_encoding, weights_location)
        predictions = NNAlign.predict(model, [nnalign_data[i] for i in test_indices], wildcard_encoding)
    elif model_type == 'padded':
        padded_input_dim = 4 + (13 + len(positions))*wildcard_encoding.shape[0]
        model = Sequential([
            Dense(56, input_dim=padded_input_dim, activation='sigmoid'),
            Dense(1, activation='sigmoid')])
        adam = optimizers.Adam()
        model.compile(adam, loss='mean_absolute_error')
        NNPad.train_padded(model, [padded_data[i] for i in train_indices], wildcard_encoding, weights_location)
        test_data = NNPad.encode_data([padded_data[i] for i in test_indices], wildcard_encoding)
        predictions= list(model.predict(test_data).flatten())
    print(len(predictions))
    print(len(test_indices))
    sys.stderr.flush()
    pearson = stats.pearsonr([undo_log(x) for x in predictions], [undo_log(cnn_data[i][2]) for i in test_indices])
    #sort the test index by their measurement
    test_indices_indices = list(range(0, len(test_indices)))
    sorted_test = sorted(test_indices_indices, key=lambda i: cnn_data[test_indices[i]][-1], reverse=True)
    sorted_prediction = sorted(test_indices_indices, key=lambda i: predictions[i], reverse=True)
    top_one_percent_test = set(sorted_test[0:int(0.05*len(sorted_test))])
    top_one_percent_predict = set(sorted_test[0:int(0.05*len(sorted_test))])
    assert len(top_one_percent_test) == len(top_one_percent_predict)
    top_fraction = len(top_one_percent_test.intersection(top_one_percent_predict))*1.0/len(top_one_percent_test)
    return (pearson, top_fraction)


    
runs = []
kf = KFold(n_splits = 5)
k = 0
for train_indices,test_indices in kf.split(cnn_data):
    runs.append([train_indices, test_indices, 'cnn', 'weights/cnn_kfold_%d' % k])
    runs.append([train_indices, test_indices, 'align', 'weights/align_kfold_%d' % k])
    runs.append([train_indices, test_indices, 'padded', 'weights/padded_kfold_%d' % k])
    k += 1

loo_alleles = ['HLA-A01:01', 'HLA-A02:01', 'HLA-A02:11', 'HLA-A23:01', 'HLA-A31:01', 'HLA-A69:01', 'HLA-B07:02', 'HLA-B08:01', 'HLA-B15:01', 'HLA-B57:01', 'HLA-B58:01']
for allele in loo_alleles:
    train_indices = []
    test_indices = []
    for i in range(0, len(alleles)):
        if alleles[i] == allele:
            test_indices.append(i)
        else:
            train_indices.append(i)
    runs.append([train_indices, test_indices, 'cnn', 'weights/cnn_loo%s' % allele])
    runs.append([train_indices, test_indices, 'align', 'weights/align_loo%s' % allele])
    runs.append([train_indices, test_indices, 'padded', 'weights/padded_loo_%s' % allele])
if __name__ == '__main__':
    with Pool(50) as p:
        performance = list(p.map(train_and_performance, runs))
        for i in range(0, len(runs)):
            print('run: %s' % runs[i][-1])
            print(performance[i])
