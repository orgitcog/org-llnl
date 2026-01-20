from typing import Union, Tuple, Optional
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor

import torch
import torch_scatter
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear, Sequential
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch.nn import Parameter
from MEAG_VAE.config import cfg


class MEAGlayer(MessagePassing):
    r"""
    One GNN layer for MEAG list.

    Args:
        in_channels (int or tuple): In Feature dimension.
        out_channels (int): Out Feature dimension.

    Returns:
        tuple: Tuple containing the output tensor and attention weights.
    """

    def __init__(self, in_channels, out_channels):
        super(MEAGlayer, self).__init__(aggr='sum')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.beta = Parameter(torch.Tensor(1))
        self._alpha = None
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin_l = Linear(in_channels[0], out_channels, bias=False)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.beta.data.fill_(1)
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        """
        Perform the forward pass of the MEAGlayer.

        Args:
            x (Tensor): Input node features.
            edge_index (Adj): Edge indices.
            size (Size, optional): Size of the output tensor. Defaults to None.

        Returns:
            Tensor: Output tensor after applying the MEAGlayer.
        """
        x1: OptPairTensor = (x, x)
        x_l = self.lin_l(x)
        src, dst = edge_index
        x_l_normalize = F.normalize(x_l, p=2., dim=0)
        alpha = torch.exp(-self.beta * torch.norm(x_l_normalize[src] - x_l_normalize[dst], dim=1))
        alpha_sum = torch_scatter.scatter(alpha, dst, dim=0, reduce='sum')
        alpha_norm = alpha / alpha_sum[dst]
        out = torch_scatter.scatter(x_l[src] * alpha_norm.view(-1, 1), dst, dim=0, reduce='sum')
        assert edge_index.shape[0] > 0, print(f"Error: edge index is empty")
        out += x_l
        return out, alpha

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Convlayer(MessagePassing):
    r"""
    A convolutional layer for building the up_list for the decoder.

    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension

    Returns:
        Tensor: Output tensor.
    """

    def __init__(self, in_channels, out_channels):
        super(Convlayer, self).__init__(aggr='sum')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = self.in_channels + (self.out_channels - self.in_channels) // 2
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.out_channels)
        )
        self.beta = Parameter(torch.Tensor(1))
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        self.lin_l0 = Linear(in_channels, out_channels, bias=False)
        self.lin_l = Linear(out_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.beta.data.fill_(1)
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index):
        if isinstance(x, Tensor):
            x1: OptPairTensor = (x, x)
        x_l = self.lin_l0(x1[0])
        src, dst = edge_index
        x_l_normalize = F.normalize(x_l, p=2., dim=0)
        alpha = torch.exp(-self.beta * torch.norm(x_l_normalize[src] - x_l_normalize[dst], dim=1))
        alpha_sum = torch_scatter.scatter(alpha, dst, dim=0, reduce='sum')
        alpha_norm = alpha / alpha_sum[dst]
        out = torch_scatter.scatter(x_l[src] * alpha_norm.view(-1, 1), dst, dim=0, reduce='sum')
        out = self.lin_l(out)
        out += x_l
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)