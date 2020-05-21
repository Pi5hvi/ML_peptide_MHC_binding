from Bio.PDB import *
from Bio.PDB.Polypeptide import is_aa
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.SubsMat import MatrixInfo as matlist
import string
import sys
import itertools
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import AlignIO
import os
import os.path
import urllib.request
from collections import *
import math
import align
import pickle
matrix = matlist.blosum62

structures_file = 'structures.txt'

class ChainKeeper(Select):
    def __init__(self, chain_ids):
        self.chain_ids = chain_ids

    def accept_chain(self, chain):
        if chain.id in self.chain_ids:
            return 1
        else:
            return 0

class ContactMapLetters:
    def __init__(self, contacts):
        self.characters = '#!$%&()*+,/0123456789;:<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^~'
        print('contact length: %d' % len(contacts))
        assert(len(contacts) <= len(self.characters))
        self.contacts = list(contacts)
    def contact_to_letter(self, contact):
        assert(contact in self.contacts)
        i = self.contacts.index(contact)
        return self.characters[i]
    def contacts_to_string(self, contacts):
        return ''.join([self.contact_to_letter(x) for x in contacts])
    def create_substitution_matrix(self):
        """
        Returns a string you should write to a file.
        """
        s = []
        s.append(' ' + self.characters[0:len(self.contacts)])
        for x in self.contacts:
            l = [self.contact_to_letter(x)]
            for y in self.contacts:
                l.append(str(len(y.intersection(x)) - 0.5*len(y.symmetric_difference(x))))
            s.append(l)
        strings = [' '.join(string) for string in s]
        return '\n'.join(strings)
            
        

def get_uniprot(uniprot_id):
    if ':' in uniprot_id:
        #then of the form UNP:P01891
        uniprot_id = uniprot_id.split(':')[1]

    filename = uniprot_id + '.xml'
    file_location = os.path.join('uniprot', filename)
    if not os.path.isfile(file_location):
        handler = urllib.request.urlretrieve('http://www.uniprot.org/uniprot/' + filename, file_location)
    record = SeqIO.read(file_location, 'uniprot-xml')
    return str(record.seq)

def get_identity_score(seq_one, seq_two):
    assert(len(seq_one) == len(seq_two))
    num_overlap = 0
    num_matches = 0
    for i in range(0, len(seq_one)):
        if seq_one[i] != '-' and seq_two[i] != '-':
            num_overlap += 1
            if seq_one[i] == seq_two[i]:
                num_matches += 1
    return (num_overlap, num_matches*1.0/num_overlap)

class PairwisePositionMapper:
    def __init__(self, reference_sequence, other_sequence):
        matrix = matlist.blosum62
        self.reference_sequence = reference_sequence
        self.other_sequence = other_sequence
        #using the default in Emboss Needle on the EBI website. Blosum62 sub matrix, gap penalty of -10, extension penalty of -0.5        
        alignment = pairwise2.align.globalds(reference_sequence, other_sequence, matrix, -10.0, -0.5, one_alignment_only=True)
        print('alignment')
        print(format_alignment(*alignment[0]))
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
    Converts position in unaligned other sequence to position in unaligned reference sequence
    """
    def get_reference_position(self, i):
        if i in self.other_to_ref_positions:
            return self.other_to_ref_positions[i]
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
    Returns two things: a list of alpha carbons for the reference chain, a list of alpha carbons for the other chain. The lists are of the same length
    """
    def alpha_carbon_lists(self, ref_chain, other_chain):
        ref_polypeptide = PPBuilder().build_peptides(ref_chain)[0]
        ref_alpha_carbons = ref_polypeptide.get_ca_list()
        other_polypeptide = PPBuilder().build_peptides(other_chain)[0]
        other_alpha_carbons = other_polypeptide.get_ca_list()
        assert(len(self.reference_matched_positions) == len(self.other_matched_positions))
        return ([ref_alpha_carbons[i] for i in self.reference_matched_positions], [other_alpha_carbons[i] for i in self.other_matched_positions])

            
    @staticmethod
    def is_polymorphic(pairwise_position_mappers, position):
        """
        Take in a list of PairwisePositionMapper objects, and a position (relative to unaligned reference sequence), and returns True if:

        1) In the alignments, the position is never a gap in the non-reference sequence

        AND

        2) At least two sequences differ in the amino acid at that position
        """
        normal_pos_aa = pairwise_position_mappers[0].get_position(position)
        for x in pairwise_position_mappers:
            pos_aa = x.get_aa(position)
            if pos_aa == '-':
                return False
            if pos_aa != normal_pos_aa:
                return True
        return True
            

class PositionMapper:
    def __init__(self, reference_sequence):
        self.mapper = None
        self.reference_sequence = reference_sequence
        #map PDB ID to sequence of the chain
        self.chain_sequences = {}
        self.polymorphic_positions = []
    def add_chain(self, pdb_id, sequence):
        self.chain_sequences[pdb_id] = sequence

    def run_alignment(self, sequences_file, clustal_output_file):
        """
        The sequences_file is the name of the FASTA file to store the sequences to
        The clustal_output_file is where Clustal Omega should write the output to
        """
        seq_records = [SeqRecord(Seq(sequence), id=key) for key, sequence in self.chain_sequences.items()] + [SeqRecord(Seq(self.reference_sequence), id='reference')]
        with open(sequences_file, 'w') as output_handler:
            SeqIO.write(seq_records, output_handler, 'fasta')
            
        clustalo = ClustalOmegaCommandline(infile=sequences_file, outfile=clustal_output_file, auto=True, verbose=True, force=True, infmt='fasta', outfmt='clustal')
        print(clustalo)
        clustalo()
        alignment = AlignIO.read(clustal_output_file, 'clustal')
        aligned_sequences = {record.id: str(record.seq) for record in alignment}
        reference_seq = aligned_sequences['reference']
        print('reference seq: %s' % reference_seq)
        assert('-' not in reference_seq.strip('-'))
        chain_counters = {id: 0 for id in aligned_sequences.keys()}
        del aligned_sequences['reference']
        self.mapper = {id: {} for id in aligned_sequences.keys()}
        for i in range(0, len(reference_seq)):
            is_polymorphic = False
            amino_acid = reference_seq[i]
            for pdb_id, sequence in aligned_sequences.items():
                assert(pdb_id is not 'reference')
                if sequence[i] is not '-':
                    self.mapper[pdb_id][i] = chain_counters['reference']
                    if sequence[i] != amino_acid:
                        is_polymorphic = True
                    chain_counters[pdb_id] += i
            if reference_seq[i] is not '-':
                if is_polymorphic:
                    self.polymorphic_positions.append(i)
                chain_counters['reference'] += 1
    def get_reference_position(self, i, pdb_id):
        return self.mapper[pdb_id][i]
    def is_polymorphic(self, position):
        return position in self.polymorphic_positions


"""
This class takes in the HLA chain of the reference (see superimpose_chain and superimpose_chain_sequence), the sequence of it, and the structure of it, and a 'moving' structure to "align" against it.

"""
class ReferenceSuperimposer:
    def __init__(self, ref_hla_chain, ref_hla_sequence, ref_structure, moving_hla_chain, moving_hla_sequence, moving_structure):
        mapper = PairwisePositionMapper(ref_hla_sequence, moving_hla_sequence)
        self.moving_structure = moving_structure
        self.ref_structure = ref_structure
        ref_atoms, moving_atoms = mapper.alpha_carbon_lists(ref_hla_chain, moving_hla_chain)
        sup = Superimposer()
        sup.set_atoms(ref_atoms, moving_atoms)
        self.sup = sup
        self.sup.apply(self.moving_structure.get_atoms())

    """
    This gives us the positions of the alpha carbons in space of the peptide. A list of 3-tuples, equal in length to the peptide. 
    """
    def get_peptide_points(self, peptide_chain_id):
        chains = {x.id: x for x in self.moving_structure.get_chains()}
        peptide_chain = chains[peptide_chain_id]
        peptide_chain_pp = PPBuilder().build_peptides(peptide_chain)[0]
        alpha_carbons = peptide_chain_pp.get_ca_list()
        return [x.get_coord() for x in alpha_carbons]

    """
    This superimposes the moving struct onto the reference struct, and returns a structure of the two combined.
    """
    def superimpose(self):
        main_structure = self.moving_structure

        i = len(list(main_structure.get_chains()))
        print('starting i: ' + str(i))
        chain_sequences = []
        added_chain_ids = []
        for chain in self.ref_structure.get_chains():
            chain_sequence = str(PPBuilder().build_peptides(chain)[0].get_sequence())
            if chain_sequence not in chain_sequences and len(chain_sequence) < 14:
                chain_sequences.append(chain_sequence)
                chain.detach_parent()
                print(chain.get_residues())
                #chain = chain.copy()
                chain.id = i#string.ascii_uppercase[i]
                added_chain_ids.append(chain.id)                
                print('chain id: %s' % chain.id)
                main_structure[0].add(chain)
                i += 1
        return (main_structure, added_chain_ids)

    

    
hla_sequence = 'GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKAHSQTDRANLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAVHAAEQRRVYLEGRCVDGLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLLGAVITGAVVAAVMWRRKSSDRKGGSYTQAASSDSAQGSDVSLTACKV'

"""
This is the structure we will superimpose on
"""

superimpose_id = '3mre'
superimpose_chain = None
superimpose_peptide_chain = None
superimpose_chain_sequence = None
superimpose_struct = None


super_peptide_chain_pp = None
super_alpha_carbons = None
super_atoms = None
super_min = None
super_max = None
 

#mapper = PositionMapper(hla_sequence)

#maps the PDB ID to a PairwisePositionMapper object
pairwise_position_mappers = {}
hla_chains = {} #map PDB id to the ID of the pocket chain
peptide_chains = {}
peptide_lengths = [8, 9, 10, 11, 12, 13]
contact_map = {i: list() for i in peptide_lengths}
peptide_length_distribution = defaultdict(int)
hla_chain_sequences = []

"""
Maps the PDB ID 
"""
structure_map = {}

"""
First, get all of the chain sequences together, and do the pairwise alignments with the reference.
"""
z = 0
with open(structures_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line.startswith('#') and len(line) > 1:
            """
            if z >= 10:
                break
            z += 1
            """
            p = PDBParser()
            pdb_id = line.lower()
            print('pdb id: ' + pdb_id)
            structure = p.get_structure('X', os.path.join('structures', 'pdb' + pdb_id + '.ent'))
            chains = structure.get_chains()
            
            best_chain_sequence = None
            best_identity_score = 0
            best_alignment = None
            chain_list = list(chains)
            peptide_chain = None
            peptide_chain_sequence = None
            peptide_length = 0
            chain_dict = {x.id: x for x in chain_list}
            chain_sequences = {}
            hla_chain = None
            manual_peptide_chain = False
            for chain in chain_list:
                sequence = str(PPBuilder().build_peptides(chain)[0].get_sequence())
                chain_sequences[chain.id] = sequence
                if len(sequence) in peptide_lengths and manual_peptide_chain is False and sequence != peptide_chain_sequence:
                    peptide_length = len(sequence)
                    if peptide_chain is not None and sequence:
                        print('pdb id: ' + pdb_id)
                        peptide_chain_id = input('What\'s the peptide chain? Here\'s a list: ' + ', '.join([x.id for x in chain_list]))
                        peptide_chain = chain_dict[peptide_chain_id]
                        peptide_chain_sequence = sequence
                        manual_peptide_chain = True
                    else:
                        peptide_chain_sequence = sequence
                        peptide_chain = chain
                alignment = pairwise2.align.localds(hla_sequence, sequence, matrix, -10, -0.5, one_alignment_only = True)
                num_overlap, identity_score = get_identity_score(alignment[0][0], alignment[0][1])
                print('identity score: %f' % identity_score)
                print('num overlap: %d' % num_overlap)
                print('Length: %d' % len(sequence))
                if num_overlap >= 200 and identity_score > best_identity_score:                    
                    best_chain_sequence = sequence
                    best_chain_id = chain.id
                    best_identity_score = identity_score
                    best_alignment = alignment
                    hla_chain = chain
            hla_chain_sequences.append(SeqRecord(best_chain_sequence, pdb_id))
            
            if peptide_chain is None:
                print('pdb id: ' + pdb_id)
                peptide_chain_id = input('What\'s the peptide chain? Here\'s a list: ' + ', '.join([x.id for x in chain_list]))
                peptide_chain = chain_dict[peptide_chain_id]
                peptide_chain_sequence = sequence
                manual_peptide_chain = True
            peptide_chains[pdb_id] = peptide_chain.id
            print(best_alignment)
            if best_identity_score < 0.80:
                print('PDB ID: ' + pdb_id)
                print('best identity score: %f' % best_identity_score)
                while True:
                    chain_id = input('What\'s the best HLA chain? Here is a list: ' + ', '.join([x.id for x in chain_list]) + ': ')                                    
                    if chain_id in chain_dict:
                        best_chain_sequence = chain_sequences[chain_id]
                        best_chain_id = chain_id
                        hla_chain = chain_dict[best_chain_id]
                        break
            pairwise_position_mappers[pdb_id] = PairwisePositionMapper(hla_sequence, best_chain_sequence)
            #mapper.add_chain(line.lower(), best_chain_sequence)
            hla_chains[pdb_id] = best_chain_id
            assert(hla_chain)
            if pdb_id == superimpose_id:
                superimpose_chain = hla_chain
                superimpose_peptide_chain = peptide_chain
                superimpose_chain_sequence = best_chain_sequence
                superimpose_struct = structure
                super_peptide_chain_pp = PPBuilder().build_peptides(superimpose_peptide_chain)[0]
                super_alpha_carbons = super_peptide_chain_pp.get_ca_list()
                super_atoms = [x.get_coord() for x in super_alpha_carbons]
                super_min = [min([x[0] for x in super_atoms]), min([x[1] for x in super_atoms]), min([x[2] for x in super_atoms])]
                super_max = [max([x[0] for x in super_atoms]), max([x[1] for x in super_atoms]), max([x[2] for x in super_atoms])]

#mapper.run_alignment('sequences.fasta', 'clustal_output_file.clustal')
"""
with open('hla_chains.fasta', 'w') as f:
    SeqIO.write(hla_chain_sequences, f, 'fasta')
"""
z = 0

assert(superimpose_chain)

peptide_contacts = {}

class ContactSelector(Select):
    def __init__(self, keep_residues):
        self.keep_residues = keep_residues
    def accept_residue(self, residue):
        if residue in self.keep_residues:
            return 1
        else:
            return 0
pdb_to_contact_count = {}

length_to_points_map = defaultdict(list)
with open(structures_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line.startswith('#') and len(line) > 1:
            """
            if z >= 10:
                break
            z += 1
            """
            p = PDBParser()
            pdb_id = line.lower()
            print('pdb id: ' + pdb_id)
            structure = p.get_structure('X', os.path.join('structures', 'pdb' + pdb_id + '.ent'))
            chains = {x.id: x for x in structure.get_chains()}
            hla_chain = chains[hla_chains[pdb_id]]
            
            peptide_chain = chains[peptide_chains[pdb_id]]
            
            peptide_sequence = str(PPBuilder().build_peptides(peptide_chain)[0].get_sequence())
            hla_sequence = str(PPBuilder().build_peptides(hla_chain)[0].get_sequence())
            sup = ReferenceSuperimposer(superimpose_chain, superimpose_chain_sequence, superimpose_struct, hla_chain, hla_sequence, structure)
            
            
            superimposed, chain_ids = sup.superimpose()
            chain_ids.append(peptide_chain.id)
            #chain_ids.append(hla_chain.id)
            io_sup = PDBIO()
            io_sup.set_structure(superimposed)
            io_sup.save(os.path.join('superimposed_structures', 'pdb' + pdb_id + '.ent'), ChainKeeper(chain_ids))
            chain_ids.append(hla_chain.id)
            io_sup.save(os.path.join('superimposed_pockets', 'pdb' + pdb_id + '.ent'), ChainKeeper(chain_ids))
            points = sup.get_peptide_points(peptide_chain.id)
            length_to_points_map[len(peptide_sequence)].append(points)
            print('points')
            print(points)
            print('pdb id: %s' % pdb_id)
            j = 0
            for point in points:
                for i in [0, 1, 2]:
                    if point[i] < super_min[i] and point[i] < super_min[i] - 4:
                        print('j: %d, i: %d' % (j, i))
                        print('super min: %f, this min: %f' % (super_min[i], point[i]))
                    if point[i] > super_max[i] and point[i] > super_max[i] + 4:
                        print('j: %d, i: %d' % (j, i))
                        print('super max: %f, this min: %f' % (super_max[i], point[i]))
                j += 1

            print(peptide_chains[pdb_id])            
            print('peptide chain id: %s' % str(peptide_chain.id))
            print('hla chain id: %s' % str(hla_chain.id))
            print('pdb id: %s' % pdb_id)
            print(', '.join(['(%f, %f, %f)' % (x[0], x[1], x[2]) for x in sup.get_peptide_points(peptide_chain.id)]) + '\n')

points_list = []
lengths = []
rep_sequences = []
for length, points in length_to_points_map.items():
    points_list.append(points)
    rep_sequences.insert(0, ''.join('A'*length))
    lengths.append(length)

data, func, references = align.make_struct(points_list)
results = align.run_alignment(func, data, 13, lengths[::-1])
print(results[0])

with open('padding.pickle', 'wb') as f:
    i = 0
    pads = []
    for x in results[0]:
        aligned = x.align(rep_sequences[i])
        print('aligned: %s' % aligned)
        assert(len(aligned) == 13)
        pads.append(aligned)
        i += 1
    pad_dict = {}
    for x in pads:
        pad_dict[len(x) - x.count('-')] = x
    print('pad dict')
    print(pad_dict)
    pickle.dump(pad_dict, f)
input('hello')    
print('lengths: ')
print(lengths[::-1])
with open(structures_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line.startswith('#') and len(line) > 1:
            """
            if z >= 10:
                break
            z += 1
            """
            p = PDBParser()
            pdb_id = line.lower()
            structure = p.get_structure('X', os.path.join('structures', 'pdb' + pdb_id + '.ent'))
            io = PDBIO()
            io.set_structure(structure)
            
            chains = {x.id: x for x in structure.get_chains()}
            hla_chain = chains[hla_chains[pdb_id]]
            hla_amino_acids = list(filter(lambda x: is_aa(x), hla_chain.get_residues()))
            hla_atoms = [x.get_atoms() for x in hla_amino_acids]
            neighbor_search = NeighborSearch(list(itertools.chain(*hla_atoms)))
            keep_residues = set()
            
            peptide_chain = chains[peptide_chains[pdb_id]]
            peptide_sequence = str(PPBuilder().build_peptides(peptide_chain)[0].get_sequence())
            """
            Now that we have both the peptide and HLA chains, and the alignment, iterate 
            """
            peptide_residues = list(filter(lambda x: is_aa(x), peptide_chain.get_residues()))
            peptide_length = len(peptide_residues)
            
            peptide_contact = []
            num_contacts = 0
            if peptide_length in peptide_lengths:
                hla_residues = list(filter(lambda x: is_aa(x), hla_chain.get_residues()))
                peptide_length_distribution[peptide_length] += 1
                i = 0
                for peptide_residue in peptide_residues:
                    keep_residues.add(peptide_residue)
                    position_contact_set = set()
                    for residue_atom in peptide_residue.get_atoms():
                        contact_residues = neighbor_search.search(residue_atom.get_coord(), 4, level='R')
                        print('contact residues')
                        print(contact_residues)
                        for contact_residue in contact_residues:
                            if is_aa(contact_residue):
                                assert(contact_residue in hla_residues)
                                position = hla_residues.index(contact_residue)
                                reference_position = pairwise_position_mappers[pdb_id].get_reference_position(position)
                                keep_residues.add(contact_residue)
                                if PairwisePositionMapper.is_polymorphic(list(pairwise_position_mappers.values()), reference_position):
                                    #print('reference position: %d' % reference_position)                        
                                    contact_map[peptide_length].append((i, reference_position))
                                    position_contact_set.add(reference_position)

                            
                    i += 1
                    peptide_contact.append(frozenset(position_contact_set))
                peptide_contacts[pdb_id] = peptide_contact
                assert(len(peptide_contact) > 0)
                io.save(os.path.join('contact_structures', 'pdb' + pdb_id + '.ent'), ContactSelector(keep_residues))
                pdb_to_contact_count[pdb_id] = len(keep_residues)
            else:
                print('peptide length: %d, peptide: %s' % (peptide_length, peptide_sequence))
                assert(False)


assert(superimpose_chain)
assert(superimpose_chain_sequence)
with open(os.path.join('contact_structures', 'counts.txt'), 'w') as f:
    contact_counts = sorted(pdb_to_contact_count.items(), key=lambda x: x[1], reverse=True)
    for x, y in contact_counts:
        f.write('PDB: %s, count: %d\n' % (x, y))
        
peptide_contact_types = set()
for pdb_id, peptide_contact in peptide_contacts.items():
    for x in peptide_contact:
        assert(isinstance(x, frozenset))
        peptide_contact_types.add(x)
print('peptide contact types: %d' % len(peptide_contact_types))



"""
cml = ContactMapLetters(peptide_contact_types)
for x in peptide_contact_types:
    print(x)
    print(cml.contact_to_letter(x))

sub_matrix = cml.create_substitution_matrix()
with open('matrix.mat', 'w') as f:
    f.write(sub_matrix)

with open('contacts.fasta', 'w') as f:    
    for pdb_id, peptide_contact in peptide_contacts.items():
        print('peptide_contact')
        print(peptide_contact)
        contacts_as_letters = cml.contacts_to_string(peptide_contact)
        f.write('>%s\n' % pdb_id)
        f.write('%s\n' % contacts_as_letters)
"""

threshold = 0.25

f = open('positions.txt', 'w')
g = open('positions_late.txt', 'w')
length_to_pocket_positions = {}
"""
Maps the peptide length to a dictionary that maps each pocket position to a corresponding peptide position.
"""
length_to_pocket_to_peptide = defaultdict(lambda: defaultdict(list))
for length in peptide_lengths:
    num_length_occurence = peptide_length_distribution[length]
    print('num length occurence: %d' % num_length_occurence)
    print('peptide length: %d' % length)
    f.write('peptide length: %d\n' % length)
    g.write('peptide length: %d\n' % length)
    contact_dict = defaultdict(list)
    for k,v in list(set(contact_map[length])):
        contact_dict[k].append(v)
    pocket_pos_list = []
    pairs = {}
    for position in range(0, length):
        contacts = []
        for x in peptide_contacts.values():
            if len(x) == length:
                contacts.extend(list(x[position]))
        contact_count = Counter(contacts)
        keep_contacts = set()
        for pocket_position,count in contact_count.items():
            if count >= threshold*num_length_occurence:
                keep_contacts.add(pocket_position)
                length_to_pocket_to_peptide[length][pocket_position].append(position)
        f.write('pep pos: %d, pocket positions: %s' % (position, ', '.join([str(x) for x in sorted(list(keep_contacts))])) + '\n')
        pocket_pos_list.extend(list(keep_contacts))
        pairs[position] = set(keep_contacts)
    pocket_positions = sorted(list(set(pocket_pos_list)))
    length_to_pocket_positions[length] = pocket_positions
    g.write(''.join(['|l']*len(pocket_positions)) + '|\n')
    g.write(' & '.join([str(x) for x in pocket_positions]) + '\n')    
    for position in range(0, length):
        line = '%d &' % position
        columns = ['\\cellcolor{black}' if i in pairs[position] else ' ' for i in pocket_positions]
        line += ' & '.join(columns)
        g.write(line + '\\ \hline\n')
length_to_pocket_to_peptide_dict = {}
for k, v in length_to_pocket_to_peptide.items():
    length_to_pocket_to_peptide_dict[k] = dict(v)
f.close()
g.close()
h = open('pocket_positions.pickle', 'wb')
pickle.dump(length_to_pocket_positions, h)
h.close()
h = open('length_to_pocket_to_peptide.pickle', 'wb')
pickle.dump(length_to_pocket_to_peptide_dict, h)
h.close()
