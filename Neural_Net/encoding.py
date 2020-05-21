from Bio import pairwise2
from Bio.Alphabet import IUPAC
from Bio import SeqIO
from Bio.SubsMat import MatrixInfo as matlist
from collections import *
import random
import pickle
import csv
import re

class PositionGetter:
    def __init__(self, pickle_location):
        with open(pickle_location, 'rb') as f:
            self.length_to_pocket_positions = dict(pickle.load(f))
    def get_pocket_positions(self, peptide):
        return self.length_to_pocket_positions[len(peptide)]
    def get_dict(self):
        return self.length_to_pocket_positions



class PositionMapper:
    def __init__(self, other_sequence):
        matrix = matlist.blosum62
        self.reference_sequence = 'GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKAHSQTDRANLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAVHAAEQRRVYLEGRCVDGLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLLGAVITGAVVAAVMWRRKSSDRKGGSYTQAASSDSAQGSDVSLTACKV'
        self.other_sequence = other_sequence
        #using the default in Emboss Needle on the EBI website. Blosum62 sub matrix, gap penalty of -10, extension penalty of -0.5        
        alignment = pairwise2.align.globalds(self.reference_sequence, other_sequence, matrix, -10.0, -0.5, one_alignment_only=True)        
        ref_alignment = alignment[0][0]
        other_alignment = alignment[0][1]
        self.ref_to_other_positions = {}
        self.other_to_ref_positions = {}
        assert(len(ref_alignment) == len(other_alignment))
        ref_position = 0
        other_position = 0
        other_started = False
        ref_started = False
        """
        self.reference_matched_positions should be the same length as self.other_matched_positions
        
        Basically, just have positions of match states in both the reference and other sequence
        """
        self.reference_matched_positions = []
        self.other_matched_positions = []
        for i in range(0, len(ref_alignment)):
            if (not ref_started) and ref_alignment[i] != '-':
                ref_started = True
            if (not other_started) and other_alignment[i] != '-':
                other_started = True
            old_ref_position = ref_position
            if other_started and ref_started and ref_alignment[i] != '-' and other_alignment[i] != '-':
                self.reference_matched_positions.append(ref_position)
                self.other_matched_positions.append(other_position)

            if ref_alignment[i] != '-' and other_started:
                self.ref_to_other_positions[ref_position] = other_position
            if ref_alignment[i] != '-':
                ref_position += 1
            if other_alignment[i] != '-' and ref_started: 
                self.other_to_ref_positions[other_position] = old_ref_position
            if other_alignment[i] != '-':
                other_position += 1

    """
    Takes unaligned position in reference, returns unaligned position in other sequence.
    """
    def get_position(self, i):
        if i in self.ref_to_other_positions:
            return self.ref_to_other_positions[i]
        else:
            return -1

    """
    Converts position in unaligned reference position to position in unaligned other sequence, then returns amino acid at that position in other sequence
    """
    def get_aa(self, i):
        pos = self.get_position(i)
        if pos >= 0:
            return self.other_sequence[pos]
        else:
            return '-'


        
"""
the dictionary returned from here maps the MHC alleles to a dictionary that contains the pseudosequence for that allele.

allele_map_file is a CSV file that maps the allele name to the code used in the FASTA file


positions_list is the list of positions that form the pseudosequence
"""
def get_pseudosequences_dict(allele_map_file, allele_fasta_file, positions_list):
    alleles = list()
    codes = list()
    alphabet = set(IUPAC.IUPACProtein.letters)
    with open(allele_map_file, 'r') as f:
        reader = csv.DictReader(f)        
        for row in reader:
            if len(row) > 0:
                allele = row['Allele']
                code = row['Code']
                alleles.append(allele)
                codes.append(code)
    assert(len(alleles) == len(codes))
    pseudosequences = {}
    print('codes')
    print(codes)
    code_extractor = re.compile('^[A-Za-z0-9]+(?::|\|)(?P<code>[A-Za-z0-9]+)')
    with open(allele_fasta_file, 'rU') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            code = code_extractor.match(record.name).group('code')
            thing = 0
            if code in codes:
                index = codes.index(code)
                allele = alleles[index]
                mapper = PositionMapper(str(record.seq))                
                pseudosequence = ''.join([mapper.get_aa(i) for i in positions_list])
                assert(set(pseudosequence).issubset(alphabet))
                pseudosequences[allele] = pseudosequence

    for x in alleles:
        if x not in pseudosequences:
            print(x)
    return pseudosequences
            
        
class Error(Exception):
    pass

class NoSuchAAError(Error):
    def __init__(self, aa):
        self.expression = str(aa)
        self.message = 'No such amino acid'

encoder_matrix = """A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0  
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2 
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2 
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3 
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2 
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 """

class BlossumEncoder:
    def __init__(self):
        """
        matrix = matlist.blosum62
        print('matrix')
        print(len(matrix.keys()))
        encoder = defaultdict(list)
        for k, v in matrix.items():
            first_aa, second_aa = k
            encoder[first_aa].append((second_aa, v))
            if first_aa != second_aa:
                encoder[second_aa].append((first_aa, v))
        print('encoder')
        
        self.encode_dict = {k: [x[1] for x in sorted(v, key=lambda x: x[0])] for k,v in encoder.items()}
        """
        encoder_lines = encoder_matrix.split('\n')
        self.encode_dict = {}
        for line in encoder_lines:
            parts = line.split()
            assert(len(parts) == 21)
            aa = parts[0]
            self.encode_dict[aa] = [float(x) for x in parts[1::]]
        assert(len(self.encode_dict.items()) == 20)
    def encode_aa(self, aa):
        if aa in self.encode_dict:
            return self.encode_dict[aa]
        else:
            raise NoSuchAAError(aa)

"""
pg = PositionGetter('pocket_positions.pickle')
positions = pg.get_pocket_positions('A'*9)
print('positions')
print(positions)
pseudosequences = get_pseudosequences_dict('mhc_mapper.csv', 'complete_mhc.fasta', positions)
print('pseudosequences')
print(pseudosequences)
print(len(pseudosequences.items()))
print(len(pg.get_pocket_positions('AAAAAAAAAA')))


encoder = BlossumEncoder()
print(encoder.encode_aa('A'))
reference_sequence = 'GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKAHSQTDRANLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAVHAAEQRRVYLEGRCVDGLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLLGAVITGAVVAAVMWRRKSSDRKGGSYTQAASSDSAQGSDVSLTACKV'


def deletion_range_consistent(ranges, new_range, essential_positions):
    #make sure we don't delete an essential_position or a something we're already planning on deleting
    for i in range(new_range[0], new_range[1]):
        for x, y in ranges:
            if i >= x and i < y:
                return False
        if i in essential_positions:
            return False
    return True

def addition_consistent(additions, new_addition):
    ranges = [(x, x + len(s)) for  x,s in additions]
    new_addition_range = (new_addition[0], new_addition[0] + len(new_addition[1]))
    return deletion_range_consistent(ranges, new_addition_range, [])
def delete(seq, deletion_ranges):
    seq_list = list(seq)
    for s,e in deletion_ranges:
        for i in range(s, e):
            seq_list[i] = ''
    return ''.join(seq_list)

def add(seq, additions):
    seq_list = list(seq)
    for start, chunk in additions:
        seq_list[start] += chunk
    return ''.join(seq_list)
def check(reference, positions, position_mapper):
    for i in positions:
        assert(position_mapper.get_aa(i) == reference[i])
        
alphabet = list(set(reference_sequence))
essential_positions = [4, 10, 20, 21, 22, 100, 103, 105]
chunk_sizes = [1, 2, 3, 4, 5, 6, 7]
num_chunks_add = [0, 1, 2, 3, 4]
random.seed(5)
num_chunks_delete = [0, 1, 2, 3, 4]
for chunk_size in chunk_sizes:
    for num_add in num_chunks_add:
        for num_delete in num_chunks_delete:
            deletion_ranges = []
            i = 0
            while i < num_delete:
                delete_start = random.randint(0, len(reference_sequence) - chunk_size)
                delete_end = delete_start + chunk_size
                if deletion_range_consistent(deletion_ranges, (delete_start, delete_end), essential_positions):
                    deletion_ranges.append((delete_start, delete_end))
                    i += 1
            deleted_seq = delete(reference_sequence, deletion_ranges)            
            i = 0
            additions = []
            for i in range(0, num_add):
                insert_seq = ''.join(random.choices(alphabet, k = chunk_size))
                insert_start = -1
                while insert_start < 0:
                    insert_start = random.randint(0, len(deleted_seq) - 1)
                    if not addition_consistent(additions, (insert_start, insert_seq)):
                        insert_start = -1
                additions.append((insert_start, insert_seq))
            add(deleted_seq, additions)
            position_mapper = PositionMapper(deleted_seq)
            check(reference_sequence, essential_positions, position_mapper)

print('hello')
"""
