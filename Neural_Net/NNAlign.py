import numpy as np
import tempfile
import math
import itertools
import sys
import random
def undo_log(x):
    return math.e**((1 - x)*math.log(50000))
"""
Returns a single numpy array, with the order:

peptide
length
pocket
"""
def core_encoding(peptide_encoding, peptide_length, pocket_encoding, core_length):
    length_encoding = np.array([1 if peptide_length <= core_length - 1 else 0, 1 if peptide_length == core_length else 0, 1 if peptide_length == core_length + 1 else 0, 1 if peptide_length > core_length + 1 else 0])
    vectors = []
    vectors.extend(peptide_encoding)
    vectors.append(length_encoding)
    vectors.extend(pocket_encoding)
    return np.concatenate(vectors)
    


def generate_cores(peptide, wildcard_encoding, core_length = 9):
    cores = []
    peptide_length = peptide.shape[0]
    if peptide_length < core_length:
        #try inserting consecutive wildcards
        for i in range(0, peptide_length + 1):
            core = np.insert(peptide, i, wildcard_encoding, axis=0)
            cores.append(core)
    elif peptide_length == core_length:
        #don't do anything
        cores.append(peptide)
    else:
        #delete consecutive in core
        deletion_window_length = peptide_length - core_length
        for i in range(0, peptide_length - deletion_window_length + 1):
            core = np.delete(peptide, np.arange(i, i + deletion_window_length), axis=0)
            cores.append(core)
        if deletion_window_length > 1:
            #delete at both end terminals
            for i in range(1, deletion_window_length):
                #i is amount we delete at end terminal                
                core = np.delete(peptide, np.concatenate((np.arange(0, deletion_window_length - i), np.arange(peptide.shape[0] - i, peptide.shape[0]))), axis=0)
                cores.append(core)
    return cores

def predict(model, data, wildcard_encoding):
    core_length = 9
    train_core_sets = [generate_cores(peptide, wildcard_encoding, core_length) for peptide,pocket,measure in data]
    train_core_lengths = [len(x) for x in train_core_sets]
    train_core_encodings = []
    i = 0
    for cores in train_core_sets:
        peptide,pocket,measure = data[i]
        pep_len = peptide.shape[0]
        train_core_encodings.extend([core_encoding(core, pep_len, pocket, core_length) for core in cores])
        i += 1
    predictions = model.predict(np.array(train_core_encodings))
    final_predictions = []
    cores_index = 0
    for length in train_core_lengths:
        #length is the # of cores generated.
        best_prediction = -1
        best_core = None
        for i in range(0, length):
            if predictions[cores_index][0] > best_prediction:
                best_core = train_core_encodings[cores_index]
                best_prediction = predictions[cores_index][0]
            cores_index += 1
        assert(best_core is not None)
        final_predictions.append(best_prediction)
    return final_predictions
def train_nnalign(model, train_set, wildcard_encoding, weights_location, core_length = 9):
    epochs = 1000
    train_core_sets = [generate_cores(peptide, wildcard_encoding, core_length) for peptide,pocket,measure in train_set]
    train_core_lengths = [len(x) for x in train_core_sets]
    train_core_encodings = []
    i = 0
    for cores in train_core_sets:
        peptide,pocket,measure = train_set[i]
        pep_len = peptide.shape[0]
        train_core_encodings.extend([core_encoding(core, pep_len, pocket, core_length) for core in cores])
        i += 1
    for x in range(0, epochs):
        predictions = model.predict(np.array(train_core_encodings))
        training_data = []
        training_targets = []
        print('got predictions')
        train_index = 0
        cores_index = 0
        for length in train_core_lengths:
            #length is the # of cores generated.
            best_prediction = -1
            best_core = None
            for i in range(0, length):
                if predictions[cores_index][0] > best_prediction:
                    best_core = train_core_encodings[cores_index]
                    best_prediction = predictions[cores_index][0]
                cores_index += 1
            assert(best_core is not None)
            #best_core = train_core_encodings[random.randint(0, length - 1)]
            training_data.append(best_core)
            training_targets.append(train_set[train_index][2])
            train_index += 1
        print('got the best cores')
        print('going to do a round of training')
        model.train_on_batch(np.array(training_data), np.array(training_targets))
    model.save_weights(weights_location)
