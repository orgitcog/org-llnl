import os
import sys
import torch
import torch.nn as nn

from models.base import mBaseModel
from pdb import set_trace


class GeomFusion(mBaseModel):

    def __init__(
        self,
        in_channels,
        out_channels,
        loss_fn,
        models={},
        attention=True,
        attention_type='fc',
        estpose='False',
        freeze=False,
    ):
        super(GeomFusion, self).__init__(in_channels, out_channels, loss_fn, estpose)
        self.first_name, self.first_model = models["first"]
        self.second_name, self.second_model = models["second"]
        self.attention = attention
        self.freeze = freeze

        dims = {
            "sgcnn": 12,
            "conv3d": 10,
            "pcn": 16,
            "efficientse3": 16,
            "nonefficientse3": 16,
            "egnn": 16,
            "feat_1": 5,
            "feat_2": 10,
        }
        self.attention_type = attention_type
        
        self.fc_first = nn.Sequential(
            nn.Linear(dims[self.first_name], dims["feat_1"]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(dims["feat_1"]),
        )

        self.fc_second = nn.Sequential(
            nn.Linear(dims[self.second_name], dims["feat_1"]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(dims["feat_1"]),
        )

        concat_dim = dims[self.first_name] + dims[self.second_name] + 2 * dims["feat_1"]

        
        self.attention_layer = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.Tanh(),
            nn.Softmax(dim=1),
        ) if self.attention else None

        self.fc_interim = nn.Sequential(
            nn.Linear(concat_dim, dims["feat_2"]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(dims["feat_2"]),
        )

        self.fc_last = nn.Sequential(
            nn.Linear(dims["feat_2"], 1)
        )

    def initialize(self):
        # TODO(june): add anything to be done before training
        if self.freeze:
            for param in self.first_model.parameters():
                param.requires_grad = False
            for param in self.second_model.parameters():
                param.requires_grad = False

    def _forward(self, data):

        # Ligand-predictor
        f1, coords_l = self.first_model._forward(data, get_feature=True)
        
        # Protein-predictor
        f2, coords_p = self.second_model._forward(data, get_feature=True)
        
        coords_l = coords_l[data.edge_index[1,:].unique()]
        
        coords_p = coords_p[data.edge_index[0,:].unique()]

        f1_fc = self.fc_first(f1)
        f2_fc = self.fc_second(f2)

        f_concat = torch.cat([f1, f2, f1_fc, f2_fc], axis=-1)
        # What if we actually use a transformer here ? And feed attention from first model to second model ?
    	# between `f1' and `f2' ?
        if self.attention:
            w_attention = self.attention_layer(f_concat)
            f1_f2_attn = torch.mul(f_concat, w_attention)

        f_interim = self.fc_interim(f1_f2_attn)
        out = self.fc_last(f_interim)

        if self.estpose:
            return out, coords_l, coords_p
        return out


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_dim, attention_dim)  # Fully connected layer for attention weights
        self.tanh = nn.Tanh()  # Tanh activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax activation function for attention weights

    def forward(self, x):
        # Input x has shape (batch_size, input_dim)

        # Compute attention weights
        attention_weights = self.softmax(self.tanh(self.W(x)))  # Shape: (batch_size, attention_dim)

        # Compute attention-weighted input
        attention_output = torch.mul(x, attention_weights)  # Element-wise multiplication
        return attention_output
