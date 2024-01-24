from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mpn import MPN
from features.featurization import BatchMolGraph


class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to("cuda:3")

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        weighter_matrix = torch.matmul(attention, V)
        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()
        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))
        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix

class CrossAttentionBlock(nn.Module):

    def __init__(self, args):

        super(CrossAttentionBlock, self).__init__()
        self.att = AttentionBlock(hid_dim = args.hidden_size, n_heads = 1, dropout=args.dropout)


    def forward(self,graph_feature, sequence_feature):
        output = self.att(graph_feature, sequence_feature,sequence_feature)

        return output
