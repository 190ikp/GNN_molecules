import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
import argparse
import os

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

def create_ij_bond_dict(mol, bond_dict):
    '''Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    '''
    ij_bond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        ij_bond_dict[i].append((j, bond))
        ij_bond_dict[j].append((i, bond))
    return ij_bond_dict

def extract_fingerprints(radius, atoms, ij_bond_dict, fingerprint_dict, edge_dict):
    '''Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    '''
    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        ij_edge_dict = ij_bond_dict

        for _ in range(radius):
            '''Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            '''
            nodes_ = []
            for i, j_edge in ij_edge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            '''Also update each edge ID considering
            its two nodes on both sides.
            '''
            ij_edge_dict_ = defaultdict(lambda: [])
            for i, j_edge in ij_edge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    ij_edge_dict_[i].append((j, edge))

            nodes = nodes_
            ij_edge_dict = ij_edge_dict_

    return np.array(nodes)

def create_dataset(args):
    from rdkit import Chem

    dataset = []

    # Load a dataset.
    with open('dataset/%s.txt' % args.dataset, 'r') as f:
        lines = f.readlines()

        for index, line in enumerate(lines, 1):
            smiles, property = line.strip('\n').split(' ')

            # Exclude the data contains '.' in its smiles.
            if '.' in smiles:
                continue

            # Create each data with the above defined functions.
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            ij_bond_dict = create_ij_bond_dict(mol, bond_dict)
            fingerprints = extract_fingerprints(args.radius, atoms, ij_bond_dict, fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)

            dataset.append((fingerprints, adjacency, molecular_size, property))

            print('\r%s: %5d/%5d' % (filename, index, len(lines)), end='')
        print()

    return dataset

def main(args):
    filename = 'dataset/%s.pth' % (args.dataset)

    if os.path.exists(filename):
        return

    print('Preprocessing the %s dataset.' % args.dataset)

    '''Initialize x_dict, in which each key is a symbol type
    (e.g., atom and chemical bond) and each value is its index.'''
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    dataset = create_dataset(args)
    property = [float(data[3]) for data in dataset]
    if len(np.unique(property)) == 2:
        print(' positive ratio %4.1f%%' % (sum(property) / len(dataset) * 100))

    if args.task == 'classification':
        dataset_train, dataset_test = train_test_split(dataset, train_size=0.8, test_size=0.2, shuffle=True, stratify=property)
    else:
        dataset_train, dataset_test = train_test_split(dataset, train_size=0.8, test_size=0.2, shuffle=True)

    N_fingerprints = len(fingerprint_dict)

    torch.save({
        'dataset_train':dataset_train,
        'dataset_test':dataset_test,
        'N_fingerprints': N_fingerprints,
        }, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classification target is a binary value (e.g., drug or not).
    # regression target is a real value (e.g., energy eV).
    parser.add_argument('--task', default='classification', choices=['classification', 'regression'])
    parser.add_argument('--dataset', default='hiv', choices=['hiv', 'photovoltaic', 'postera'])
    parser.add_argument('--radius', default=1)
    args = parser.parse_args([])
    print(vars(args))

    main(args)
