from random import shuffle
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from .data import MoleculeDataset, MoleculeDatapoint, MoleculeDataLoader


def filter_invalid_smiles(data):

    return MoleculeDataset([datapoint for datapoint in tqdm(data)
                            if all(s != '' for s in datapoint.smiles) and all(m is not None for m in datapoint.mol)
                            and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol if not isinstance(m, tuple))
                            and all(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() > 0 for m in datapoint.mol if isinstance(m, tuple))])


def split_data(data, sizes):

    indices = list(range(len(data)))
    shuffle(indices)

    train_size = int(sizes[0] * len(data))

    train = [data[i] for i in indices[:train_size]]
    test = [data[i] for i in indices[train_size:]]

    return MoleculeDataset(train), MoleculeDataset(test)


def get_data(path):
    data = pd.read_csv(path)

    smiles = [[smile] for smile in data["smile"].values]
    seqs = [[seq] for seq in data["seq"].values]
    labels = [[label] for label in data["labels"].values]
    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=smiles,
            sequences=sequences,
            targets=targets,
            row=None,
            data_weight=1.,
            features_generator=None,
            features=None,
            atom_features=None,
            atom_descriptors=None,
            bond_features=None,
            overwrite_default_atom_features=False,
            overwrite_default_bond_features=False
        ) for i, (smiles, sequences, targets) in tqdm(enumerate(zip(smiles, seqs, labels)))
    ])

    data = filter_invalid_smiles(data)
    data.reset_features_and_targets()
    train_data, test_data = split_data(data=data, sizes=(0.8, 0.2))
    return train_data, test_data
