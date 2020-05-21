import random
import encoding
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np
from keras.utils import plot_model
import csv
import pickle
import NNAlign
from collections import defaultdict

pg = encoding.PositionGetter('pocket_positions.pickle')
positions = pg.get_pocket_positions('A'*9)
print('positions')
print(positions)

pseudosequences = encoding.get_pseudosequences_dict('mhc_mapper.csv', 'complete_mhc.fasta', positions)
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


encoded_pseudosequences = {k: encode_sequence(v) for k,v in pseudosequences.items()}
train = []
targets = []
random.seed(1)
with open('ic50.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for x in reader:
        peptide = x['Peptide']
        encoded_peptide = encode_sequence(peptide)
        measure = float(x['1-log50k'])
        allele = x['Allele']
        if allele in encoded_pseudosequences:
            encoded_allele = encoded_pseudosequences[allele]
            partition = int(x['Partition'])
            if len(peptide) == 9:
                train.append(NNAlign.core_encoding(encoded_peptide, 9, encoded_allele, 9))
                targets.append(measure)
            if partition > 1:
                break

#train = train[0:100]
#test = test[0:100]
print('training size: %d' % len(train))



wildcard_encoding = np.array([0.0]*len(encode_sequence('A')[0]))
print('wildcard encoding shape: %d' % wildcard_encoding.shape[0])
print('num positions: %d' % len(positions))
print('pocket: %s' % pseudosequences['BoLA-D18.4'])
input_dim = 4 + (9 + len(positions))*wildcard_encoding.shape[0]
#input_dim = 4 + 9*wildcard_encoding.shape[0]
print('input dim: %d' % input_dim)
model = Sequential([
    Dense(56, input_dim=input_dim, activation='sigmoid'),
    Dense(1, activation='sigmoid')])
adam = optimizers.Adam(lr=0.0001)
model.compile(adam, loss='mean_absolute_error')
print('train')

model.fit(np.array(train), np.array(targets), epochs=1000)

plot_model(model, to_file='model.png')
model.summary()
