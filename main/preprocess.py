import numpy as np
from collections import defaultdict

def create_atoms(mol, atom_dict):
    '''Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    '''
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol, bond_dict):
    '''Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    '''
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    '''Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    '''

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            '''Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            '''
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            '''Also update each edge ID considering
            its two nodes on both sides.
            '''
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)

def split_dataset(dataset, ratio):
    '''Shuffle and split a dataset.'''
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]

def create_datasets(task, dataset, radius):

    dir_dataset = '../dataset/' + task + '/' + dataset + '/'

    '''Initialize x_dict, in which each key is a symbol type
    (e.g., atom and chemical bond) and each value is its index.
    '''
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    def create_dataset(filename):
        from rdkit import Chem

        '''Load a dataset.'''
        with open(dir_dataset + filename, 'r') as f:
            smiles_property = f.readline().strip().split()
            data_original = f.read().strip().split('\n')

        '''Exclude the data contains '.' in its smiles.'''
        data_original = [data for data in data_original if '.' not in data.split()[0]]

        dataset = []

        for index, data in enumerate(data_original, 1):

            smiles, property = data.strip().split()

            '''Create each data with the above defined functions.'''
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)

            dataset.append((fingerprints, adjacency, molecular_size, property))

            print('\r%s: %d/%d' % (filename, index, len(data_original)), end='')
        print('')
        return dataset

    dataset_train = create_dataset('data_train.txt')
    dataset_train, dataset_test = split_dataset(dataset_train, 0.8)

    N_fingerprints = len(fingerprint_dict)

    return dataset_train, dataset_test, N_fingerprints
