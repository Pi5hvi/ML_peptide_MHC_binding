from Bio.Alphabet import IUPAC
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
import cnn_copy
import tensorflow as tf

#tf.enable_eager_execution()
#tf.executing_eagerly() 

from collections import defaultdict

pg = encoding.PositionGetter('pocket_positions.pickle')
pocket_dict = pg.get_dict()
pocket_positions = list()
for v in pocket_dict.values():
    pocket_positions.extend(v)


length_to_pocket_to_peptide = None
with open('length_to_pocket_to_peptide.pickle', 'rb') as f:
    length_to_pocket_to_peptide = pickle.load(f)
unique_pocket_positions = list(set(pocket_positions))
pseudosequences = encoding.get_pseudosequences_dict('mhc_mapper.csv', 'complete_mhc.fasta', unique_pocket_positions)
blencoder = encoding.BlossumEncoder()




train = []
targets = []
random.seed(1)
with open('ic50.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for x in reader:
        peptide = x['Peptide']
        measure = float(x['1-log50k'])
        allele = x['Allele']
        if allele in pseudosequences and len(peptide) <= 13:
            train.append((peptide, pseudosequences[allele], measure))


#train = train[0:200]
#test = test[0:100]
print('training size: %d' % len(train))

hidden_layer_size = 56
print(train[0])
input('hello')
#just using the training set as the validation set for now
cnn_copy.train_cnn(hidden_layer_size, train, unique_pocket_positions, length_to_pocket_to_peptide, 'cnn.weights')
