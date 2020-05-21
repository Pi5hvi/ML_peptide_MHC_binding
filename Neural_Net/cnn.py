from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, concatenate
from keras.utils import conv_utils
from keras.layers import Layer
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
def add_lists(list_one, list_two):
    assert len(list_one) == len(list_two)
    return [list_one[i] + list_two[i] for i in range(0, len(list_one))]

class PeptidePocketConvLayer(Layer):
    def __init__(self, filter_size, alphabet, pocket_positions, peptide_length_to_pocket_to_peptide, **kwargs):
        """
        peptide_length_to_pocket_to_peptide maps the peptide length to a dict that maps each pocket position to a list of peptide positions
        """
        self.filter_size = filter_size
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.pocket_positions = list(sorted(pocket_positions))
        self.peptide_length_to_pocket_to_peptide = {k: {self.pocket_positions.index(i): v for i, v in peptide_length_to_pocket_to_peptide[k].items() } for k in peptide_length_to_pocket_to_peptide.keys()}
        #basically, 
        super(PeptidePocketConvLayer, self).__init__(**kwargs)
    def build(self, input_shape):#num_pocket_positions, filter_size, alphabet_size, max_peptide_length):
        #input_shape = [(max_peptide_length, alphabet_size, ), (num_pocket_positions, alphabet_size, )]
        #self.num_pocket_positions = num_pocket_positions
        assert isinstance(input_shape, list)
        print('input shape')
        print(input_shape)
        assert len(input_shape) == 2
        assert len(input_shape[0])== 2
        assert len(self.alphabet) == input_shape[0][1]
        self.max_peptide_length = input_shape[0][0]
        
        num_pocket_positions = input_shape[1][0]
        filter_shape = (self.filter_size,)
        self.kernel = [self.add_weight(name=str(i), shape=filter_shape, initializer='uniform', trainable=True) for i in range(0, len(self.alphabet))]
        super(PeptidePocketConvLayer, self).build(input_shape)
        
    def call(self, x):
        print('in call')        
        assert isinstance(x, list)
        pocket_encoding = x[1]
        print('pocket encoding')
        print(pocket_encoding)

        peptide_encoding = x[0]
        print('peptide encoding')
        print(peptide_encoding)
        
        print('pocket encoding')
        peptide_length = 0
        
        while peptide_length < self.max_peptide_length and any(peptide_encoding[peptide_length]):
            peptide_length += 1
            
        #assert len(pocket_encoding) == len(self.pocket_positions)
        #assert self.input_shape[1][0] == len(self.pocket_positions)
        output = []
        #size of the output vector for each pocket position
        pocket_position_output_size = self.filter_size + self.alphabet_size - 1
        for i in range(0, len(self.pocket_positions)):
            if i in self.peptide_length_to_pocket_to_peptide[peptide_length]:
                aa_filter = self.kernel[pocket_encoding[i]]
                total = np.array([0.]*pocket_position_output_size)
                for j in self.peptide_length_to_pocket_to_peptide[peptide_length][i]:
                    #j is the peptide position
                    print('peptide encoding')
                    print(peptide_encoding[j].numpy())
                    print('aa filter')
                    print(aa_filter.numpy())
                    result = np.convolve(peptide_encoding[j].numpy(), aa_filter.numpy())
                    print('result')
                    print(result)
                    total += result
                #total = np.array([i]*total.size)
                output.append(total)
            else:
                output.append([0]*pocket_position_output_size)
        return output

    def compute_output_shape(self, input_shape):
        return (len(self.pocket_positions), self.filter_size + self.alphabet_size - 1)

class SingleInputPeptidePocketConvLayer(tf.keras.layers.Layer):
    def __init__(self, filter_size, alphabet, pocket_positions, peptide_length_to_pocket_to_peptide, aa_rep_size, max_peptide_size,  **kwargs):
        
        """
        peptide_length_to_pocket_to_peptide maps the peptide length to a dict that maps each pocket position to a list of peptide positions
        """
        self.filter_size = filter_size
        self.alphabet = alphabet
        self.aa_rep_size = aa_rep_size
        self.max_peptide_length = max_peptide_size
        self.alphabet_size = len(alphabet)
        self.pocket_positions = list(sorted(pocket_positions))
        self.peptide_length_to_pocket_to_peptide = {k: {self.pocket_positions.index(i): v for i, v in peptide_length_to_pocket_to_peptide[k].items() } for k in peptide_length_to_pocket_to_peptide.keys()}
        #basically, 
        super(SingleInputPeptidePocketConvLayer, self).__init__(**kwargs)
    def build(self, input_shape):        
        num_pocket_positions = len(self.pocket_positions)
        filter_shape = (self.filter_size,)
        self.kernel = [self.add_weight(name=str(i), shape=filter_shape, initializer='uniform', trainable=True) for i in range(0, len(self.alphabet))]
        super(SingleInputPeptidePocketConvLayer, self).build(input_shape)
        
    def call(self, x):
        print('x')
        print(tf.executing_eagerly())
        
        def handle_sample(sample):
            print('eager')
            print(tf.executing_eagerly())
            print('sample')
            print(sample.numpy())

            pep_length = sample[0].numpy[0]
            print('pep length')
            print(pep_length)
            pocket_to_peptide = self.peptide_length_to_pocket_to_peptide[pep_length]
            print(pep_length)
            peptide_encoding = tf.slice(sample, [1], [self.aa_rep_size*self.max_peptide_length])
            print(peptide_encoding)
            print('hello')
            return 0
        return tf.map_fn(lambda thing: tf.py_function(handle_sample, inp=[thing], Tout=tf.int32), x)
        """
        pocket_position_output_size = self.filter_size + self.alphabet_size - 1
        
        x = K.conv1d(peptide_encoding[0], self.kernel[0])

        for i in range(0, len(self.pocket_positions)):
            if i in self.peptide_length_to_pocket_to_peptide[peptide_length]:
                aa_filter = self.kernel[pocket_encoding[i]]
                total = np.array([0.]*pocket_position_output_size)
                for j in self.peptide_length_to_pocket_to_peptide[peptide_length][i]:
                    #j is the peptide position
                    print('peptide encoding')
                    print(peptide_encoding[j].numpy())
                    print('aa filter')
                    print(aa_filter.numpy())
                    result = np.convolve(peptide_encoding[j].numpy(), aa_filter.numpy())
                    print('result')
                    print(result)
                    total += result
                #total = np.array([i]*total.size)
                output.append(total)
            else:
                output.append([0]*pocket_position_output_size)
        return output
        """

    def compute_output_shape(self, input_shape):
        return (len(self.pocket_positions), self.filter_size + self.aa_rep_size - 1)

    

def train_cnn(hidden_layer_size, train_set, validation_set, pocket_positions, peptide_length_to_pocket_to_peptide, sequence_encoder ,weights_location, filter_size, alphabet, max_peptide_length):
    train_x = []
    train_y = []
    rep_size = 0
    for peptide,pocket,measure in train_set:
        encoded_peptide = sequence_encoder(peptide)
        if rep_size == 0:
            rep_size = encoded_peptide[0].shape[0]
        array = np.array(encoded_peptide)
        assert array.shape[0] == 13
        assert array.shape[1] == 20
        flat = np.concatenate((np.array([len(peptide)]), np.ndarray.flatten(encoded_peptide)))
        pocket_array = np.array([alphabet.index(i) for i in pocket])
        concated = np.concatenate((flat, pocket_array))
        train_x.append(concated)
        train_y.append(measure)
    assert rep_size > 0
    """
    val_x_pocket = []
    val_y = []
    for peptide,pocket,measure in validation_set:
        encoded_peptide = sequence_encoder(peptide)
        val_x_peptide.append(np.array(encoded_peptide))
        val_x_pocket.append(np.array([alphabet.index(i) for i in pocket]))
        val_y.append(measure)
"""
    stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    print('shapes')
    conv_layer = SingleInputPeptidePocketConvLayer(filter_size, alphabet, pocket_positions, peptide_length_to_pocket_to_peptide, len(alphabet), 13)
    #input_shape = [(max_peptide_length, rep_size), (len(pocket_positions),)]
    input_shape = (1 + max_peptide_length*rep_size + len(pocket_positions,))
    conv_layer.build(input_shape)
    model = tf.keras.Sequential([
        conv_layer,
        tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    #optimizer = optimizers.Adam()
    model.compile(optimizer, loss='mean_absolute_error')
    print('done compiling')
    model.fit(np.array(train_x), np.array(train_y), epochs=10)
    print('done fitting')
    plot_model(model, to_file='cnn.png')
    model.summary()
    model.save_weights(weights_location)
"""
alphabet = 'ABC'
filter_size = 3
pocket_positions = [1, 5, 10, 12, 15]
peptide_length_to_pocket_to_peptide = {3: {1: [0], 5: [0, 1], 10: [1, 2], 15: [2]}, 4: {1: [0], 5: [0, 2], 10: [0, 2], 12: [3], 15:[3]}}
conv_layer = PeptidePocketConvLayer(filter_size, alphabet, pocket_positions, peptide_length_to_pocket_to_peptide)
input_shape = [(4, 3), (len(pocket_positions), 3)]
conv_layer.build(input_shape)
print(conv_layer.kernel)
for x in conv_layer.kernel:
    tf.print(x)

one_hot_encoder = {'A' : [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1]}
peptide = [one_hot_encoder['C'], one_hot_encoder['A'], one_hot_encoder['B'], [0, 0, 0]]
pocket = [one_hot_encoder['A'], one_hot_encoder['C'], one_hot_encoder['A'], one_hot_encoder['B'], one_hot_encoder['A']]
print(conv_layer.call([K.variable(np.array(peptide)), K.variable(np.array(pocket))]))
"""
