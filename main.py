import argparse

import numpy as np

import torch
from torch import nn

from data.util import *
from tape import TAPETokenizer
from models.model import InteractionModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Kd', help='dataset name: Kd/Ki/IC50')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=300, help='embedding size')
parser.add_argument('--init_lr', type=float, default=1e-4)
parser.add_argument('--max_lr', type=float, default=1e-3)
parser.add_argument('--final_lr', type=float, default=1e-4)
parser.add_argument('--sequence_length', type=int, default=2600)
parser.add_argument('--learn_rate', type=float, default=0.0025)
parser.add_argument('--tau', type=Tuple[float, float], default=(0.0 , 5.0))
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=int, default=5)
parser.add_argument('--vocab_size', type=int, default=31)
parser.add_argument('--prot_hidden', type=int, default=128)
parser.add_argument('--prot_1d_out', type=int, default=64)
parser.add_argument('--prot_1dcnn_num', type=int, default=3)
parser.add_argument('--kernel_size', type=int, default=7)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--device', type=int, default=3)
parser.add_argument('--number_of_molecules', type=int, default=1)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--ffn_num_layers', type=int, default=2)
parser.add_argument('--warmup_epochs', type=float, default=2.0)
parser.add_argument('--num_lrs', type=int, default=1)

args = parser.parse_args()
device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
tokenizer = TAPETokenizer(vocab='unirep')

def train_test(train_data, test_data):
    features_scaler = train_data.normalize_features(replace_nan_token=0)
    test_data.normalize_features(features_scaler)

    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size
    )

    loss_fn = nn.MSELoss(reduction='mean')
    model = InteractionModel(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0

        for batch in tqdm(train_data_loader):
            model.train()
            mol_batch, features_batch, target_batch, protein_sequence_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, data_weights_batch = \
                batch.batch_graph(), batch.features(), batch.targets(), batch.sequences(), batch.atom_descriptors(), \
                batch.atom_features(), batch.bond_features(), batch.data_weights()
            mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
            mask_weight = torch.Tensor([[args.alpha if list(args.tau)[0]<=x<= list(args.tau)[1] else args.beta for x in tb] for tb in target_batch])
            targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

            target_weights = torch.ones_like(targets)
            data_weights = torch.Tensor(data_weights_batch).unsqueeze(1)

            model.zero_grad()
            dummy_array = [0] * args.sequence_length

            sequence_2_ar = [list(tokenizer.encode(list(t[0]))) + dummy_array for t in protein_sequence_batch]
            new_ar = []

            for arr in sequence_2_ar:
                while len(arr) > args.sequence_length:
                    arr.pop(len(arr)-1)
                new_ar.append(np.zeros(args.sequence_length)+np.array(arr))

            sequence_tensor = torch.LongTensor(new_ar)

            preds = model(mol_batch, sequence_tensor,features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
            mask = mask.to(args.device)
            mask_weight = mask_weight.to(args.device)
            targets = targets.to(args.device)

            target_weights = target_weights.to(args.device)
            data_weights = data_weights.to(args.device)

            loss = loss_fn(preds, targets) * target_weights * data_weights * mask_weight
            loss = loss.sum() / mask.sum()

            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        results = test_model(model, test_data_loader)
        print(f'epoch: {epoch}; result: {results}')


def test_model(model, data_loader):
    actuals, preds = [], []

    model.eval()
    for batch in tqdm(data_loader):
        mol_batch, features_batch, target_batch, protein_sequence_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, data_weights_batch = \
                batch.batch_graph(), batch.features(), batch.targets(), batch.sequences(), batch.atom_descriptors(), \
                batch.atom_features(), batch.bond_features(), batch.data_weights()
        model.zero_grad()
        dummy_array = [0] * args.sequence_length

        sequence_2_ar = [list(tokenizer.encode(list(t[0]))) + dummy_array for t in protein_sequence_batch]
        new_ar = []

        for arr in sequence_2_ar:
            while len(arr) > args.sequence_length:
                arr.pop(len(arr)-1)
            new_ar.append(np.zeros(args.sequence_length)+np.array(arr))

        sequence_tensor = torch.LongTensor(new_ar)

        pred = model(mol_batch, sequence_tensor,features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
        preds.extend(pred.detach().cpu().numpy().flatten().tolist())
        actuals.extend([item[0] for item in batch.targets()])
    return get_metrics(actuals, preds)


def get_metrics(test_targets, test_preds):
    # print(f'actuals: {test_targets}\npreds: {test_preds}')
    # test_targets = np.array(actuals).flatten()
    # test_preds = np.array(preds).flatten()

    mse = mean_squared_error(test_targets, test_preds)
    rmse = np.sqrt(mse)

    pcc, _ = pearsonr(test_targets, test_preds)
    scc, _ = spearmanr(test_targets, test_preds)

    conindex = concordance_index(test_targets, test_preds)

    result = {
        "RMSE": rmse,
        'PCC': pcc,
        'SCC': scc,
        'conindex': conindex
    }
    return result


if __name__ == "__main__":
    root_path = "dataset/"

    print(f'handle the data...')
    train_data, test_data = get_data(root_path + "/" + args.dataset + ".csv")
    args.train_data_size = len(train_data)

    print(f'train the model...')
    train_test(train_data, test_data)
