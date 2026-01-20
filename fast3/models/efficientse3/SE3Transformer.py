import sys

import numpy as np
import torch

from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from .equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from .equivariant_attention.fibers import Fiber
import warnings
from pdb import set_trace

class SE3Transformer(nn.Module):
	"""SE(3) equivariant GCN with attention"""
	def __init__(self, 
			  num_layers: int, 
			  atom_feature_size: int, 
			  fiber_mid=None, 
			  fiber_out=None,
			  edge_dim: int=4, 
			  div: float=4, 
			  n_heads: int=1, 
			  dropout=0., 
			  reduce_to_scaler=True, 
			  fiber_in=None, 
			  pair_bias=True, 
			  uniform_attention=False, 
			  recycle=0, 
			  grad_through_basis=False, 
			  x_ij='add', 
			  save_params=False, 
			  share_filter=False, 
			  last_edge_update=True, 
			  **kwargs
		):
		super().__init__()
		# Build the network
		self.num_layers = num_layers
		self.edge_dim = edge_dim
		self.div = div
		self.n_heads = n_heads
		self.dropout = dropout
		self.pair_bias = pair_bias
		self.reduce_to_scaler = reduce_to_scaler
		self.recycle = recycle
		self.grad_through_basis = grad_through_basis
		self.x_ij = x_ij
		self.uniform_attention = uniform_attention
		self.save_params = save_params
		self.share_filter = share_filter
		self.last_edge_update = last_edge_update
		self.fiber_mid = Fiber(dictionary=fiber_mid)
		self.fiber_out = fiber_out
		self.fibers = {
			'in': Fiber(1, atom_feature_size) if not fiber_in else fiber_in,
			'mid': self.fiber_mid,
			'out': self.fiber_out
		}
		self.Gblock = self._build_gcn(self.fibers, 1, self.recycle)

	def _build_gcn(self, fibers, out_dim, recycle):
		Gblock = []
		fin = fibers['in']
		for i in range(self.num_layers-1):
			Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads, dropout=0, pair_bias=self.pair_bias, uniform_attention=self.uniform_attention, x_ij=self.x_ij, save_params=self.save_params, share_filter=self.share_filter))
			Gblock.append(GNormSE3(fibers['mid']))
			Gblock.append(edge_update(self.edge_dim, self.fiber_mid.structure_dict[0], dropout=self.dropout))
			fin = fibers['mid']
		Gblock.append(GSE3Res(fin, fibers['out'], edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads, dropout=self.dropout, pair_bias=self.pair_bias, uniform_attention=self.uniform_attention, x_ij=self.x_ij, save_params=self.save_params, share_filter=self.share_filter))
		if 0 in self.fiber_out.structure_dict and self.last_edge_update:
			Gblock.append(edge_update(self.edge_dim, self.fiber_out.structure_dict[0], dropout=self.dropout))
		return nn.ModuleList(Gblock)

	def forward(self, G, h=None, basis=None, r=None):
		# Compute equivariant weight basis from relative positions
		if basis == None or r == None:
			basis, r = get_basis_and_r(G, max(self.fiber_mid.structure_dict.keys()), compute_gradients=self.grad_through_basis)
		# encoder (equivariant layers)
		if not h:
			h = {'0': G.ndata['f']}
		for block in self.Gblock:
			h = block(h, G=G, r=r, basis=basis)
		return h, G.edata['w']

class edge_update(nn.Module):
	def __init__(self, edge_dim, node_dim, dropout=0.):
		super().__init__()
		self.edge_nn = nn.Sequential(
			nn.LayerNorm(edge_dim + 2 * node_dim),
			nn.Dropout(dropout), nn.Linear(edge_dim + 2 * node_dim, edge_dim), nn.LeakyReLU(), 
			nn.Dropout(dropout), nn.Linear(edge_dim, edge_dim)
		)

	def forward(self, h, G, r, basis):
		node_feat = h['0'].squeeze(-1)
		i, j = G.edges()
		x_i, x_j = node_feat[i], node_feat[j]
		e = G.edata['w']
		G.edata['w'] = e + self.edge_nn(torch.cat([x_i, x_j, e], dim=-1))
		return h

def zero_init_weights(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.zeros_(m.weight)
		m.bias.data.fill_(0)
