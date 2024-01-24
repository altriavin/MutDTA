from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mpn import MPN
from models.CAB import CrossAttentionBlock as CAB


def initialize_weights(model):
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

class InteractionModel(nn.Module):

    def __init__(self, args, featurizer=None):
        super(InteractionModel, self).__init__()

        self.featurizer = featurizer

        self.embedding_xt = nn.Embedding(args.vocab_size, args.prot_hidden)
        self.conv_in = nn.Conv1d(in_channels=args.sequence_length, out_channels=args.prot_1d_out, kernel_size=1)
        self.convs = nn.ModuleList([nn.Conv1d(args.prot_hidden, 2*args.prot_hidden, args.kernel_size, padding=args.kernel_size//2) for _ in range(args.prot_1dcnn_num)])   # convolutional layers
        self.rnns = nn.ModuleList([nn.GRU(args.prot_1d_out, args.prot_1d_out, num_layers=1, bidirectional=True) for _ in range(args.prot_1dcnn_num)])
        self.fc1_xt = nn.Linear(args.prot_hidden*args.prot_1d_out, args.hidden_size)
        self.fc_mg = nn.Linear(2048, args.hidden_size)
        self.fc_residual_connection = nn.Linear(args.prot_hidden,args.prot_1d_out)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(args.prot_1d_out)
        self.do = nn.Dropout(args.dropout)
        self.device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
        self.scale = torch.sqrt(torch.FloatTensor([args.alpha])).to(self.device)

        self.CAB = CAB(args)

        self.output_size = 1

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args):
        self.encoder = MPN(args)

    def create_ffn(self, args):

        first_linear_dim = args.hidden_size * args.number_of_molecules
        args.ffn_hidden_size = 300
        dropout = nn.Dropout(args.dropout)
        activation = nn.ReLU()

        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])


        self.ffn = nn.Sequential(*ffn)


    def featurize(self, batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch):

        return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))

    def fingerprint(self, batch, features_batch, atom_descriptors_batch):

        return self.encoder(batch, features_batch, atom_descriptors_batch)


    def normalization(self,vector_present,threshold=0.1):

        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1,keepdim = True)[0]
        de = vector_present_clone.max(1,keepdim = True)[0] - vector_present_clone.min(1,keepdim = True)[0]

        return num / de

    def forward(self, batch, sequence_tensor, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch):

        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch,
                                  atom_features_batch, bond_features_batch)
        mpnn_out = self.normalization(self.encoder(batch, features_batch, atom_descriptors_batch,
                                       atom_features_batch, bond_features_batch))

        sequence = sequence_tensor.to(self.device)
        embedded_xt = self.embedding_xt(sequence)
        input_nn = self.conv_in(embedded_xt)

        conv_input = input_nn.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):
            conved = self.norm(conv(conv_input))
            conved = F.glu(conved, dim=1)
            conved = conved + self.scale * conv_input
            conv_input = conved

        out_conv = self.relu(conved)
        protein_tensor = out_conv.view(out_conv.size(0),out_conv.size(1)*out_conv.size(2))
        protein_tensor = self.do(self.relu(self.fc1_xt(self.normalization(protein_tensor))))
        output = self.CAB(mpnn_out, protein_tensor)
        output = self.ffn(output)

        return output
