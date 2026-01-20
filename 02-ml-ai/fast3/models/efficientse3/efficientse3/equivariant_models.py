import os
import torch
import dgl
from .base_model import BaseModel
from torch_geometric.nn import global_add_pool, global_mean_pool, GlobalAttention
from torch_geometric.utils import dropout_adj
from .base_model import BaseModel
from torch.nn import Sequential, Linear, LeakyReLU, Dropout, LayerNorm, Identity, Softplus, Sigmoid
import torch.nn as nn
from .SE3Transformer import SE3Transformer
from .equivariant_attention.modules import G1x1SE3, GSE3Res
from scipy.spatial.transform import Rotation
import math
from math import pi as PI
from torch_scatter import scatter_mean
from .equivariant_attention.fibers import Fiber
from . import utils
import torch.nn.functional as F
from torch_scatter import scatter
import wandb
import copy
import numpy as np
from .base_model import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class EfficientSE3TransformerLBA(BaseModel):
	def __init__(self, in_dim, out_dim, edge_dim, output_dir, use_gpu=True, optimizer=None, schedule_lr=False,
		num_convs=3, fiber_mid=None, heads=1, out_multiplicity=32, dropout=0.,
		reduce_type='edge', reduce_from='ligand-protein', pool_method='add',
		uniform_attention=True, pair_bias=True, x_ij='add', save_params=False, share_filter=False,
		distance_bias=False, distance_cutoff=1e6, use_mse=False, atom3d=False):

		import ipdb
		ipdb.set_trace()

		self.num_convs = num_convs
		self.edge_dim = edge_dim
		self.heads = heads
		self.dropout = dropout
		self.reduce_type = reduce_type
		self.reduce_from = reduce_from
		self.pool_method = pool_method
		self.x_ij = x_ij
		self.pair_bias = pair_bias
		self.uniform_attention = uniform_attention
		self.distance_cutoff = distance_cutoff
		self.save_params = save_params
		self.share_filter = share_filter
		self.out_multiplicity = out_multiplicity
		self.distance_bias = distance_bias
		self.use_mse = use_mse
		self.fiber_mid = fiber_mid
		self.atom3d = atom3d
		super().__init__(in_dim, out_dim, output_dir, use_gpu, optimizer, schedule_lr, resume=True)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
		self.mse = nn.MSELoss()
		self.pair_distance = torch.nn.PairwiseDistance(p=2)
		if self.use_mse:
			print('Using MSE')

	def define_modules(self):
		self.interaction_linear_edge = Linear(1, self.edge_dim)
		self.linear_node = Linear(self.in_dim, 16)
		self.se3_interaction = SE3Transformer(fiber_mid=self.fiber_mid, fiber_out=Fiber(dictionary={0: self.out_multiplicity}), num_layers=self.num_convs, atom_feature_size=16, edge_dim=self.edge_dim, div=2, n_heads=self.heads, dropout=self.dropout, pair_bias=self.pair_bias, uniform_attention=self.uniform_attention, save_params=self.save_params, share_filter=self.share_filter, x_ij=self.x_ij)
		if self.reduce_type == 'node':
			hidden_dim = self.se3_interaction.fibers['out'].n_features
		else:
			hidden_dim = self.edge_dim
		self.pool = global_add_pool
		if self.pool_method == 'mean':
			self.pool = global_mean_pool
		if self.pool_method == 'attention':
			self.attn_nn = Sequential(LayerNorm(hidden_dim), Dropout(self.dropout), Linear(hidden_dim, hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(hidden_dim, 1))
			self.value_nn = Sequential(LayerNorm(hidden_dim), Dropout(self.dropout), Linear(hidden_dim, hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(hidden_dim, hidden_dim))
			self.pool = GlobalAttention(self.attn_nn, self.value_nn)
		self.out_mlp = Sequential(
			Dropout(self.dropout), Linear(hidden_dim, hidden_dim), LeakyReLU(),
			Dropout(self.dropout), Linear(hidden_dim, self.out_dim)
		)

	def forward(self, data):

		import ipdb
		ipdb.set_trace()

		G = self.make_graph(data)
		aff = self.pred_affinity(G) # (B,)
		return aff.view(-1, 1)

	def make_graph(self, data):
		G = self.ptg_to_dgl(data, distance_cutoff=self.distance_cutoff).to(self.device) # 2B grasphs
		return G

	def pred_affinity(self, G):
		# Critic: whatever structures -> affinity
		# fill in edge features
		if self.atom3d:
			edge_feat = self.edge_attr
		else:
			edge_feat = 1 / self.distances
		G.edata['w'] = self.interaction_linear_edge(edge_feat)
		G.ndata['f'] = self.linear_node(G.ndata['f'].squeeze(-1)).unsqueeze(-1)
		# compute affinity on observed and sampled states
		affinity = self.graph_to_affinity(G) # (2B)
		return affinity

	def graph_to_affinity(self, G):
		h, edge_feat = self.se3_interaction(G)
		# get invariant node features
		h = h['0'].squeeze(-1) # (N, C*D, 1) --> (N, C*D)
		affinity = self.reduce(self.x, h, self.edge_index, edge_feat, self.batch, self.distances) # (B)
		return affinity.view(-1)

	def loss(self, aff_pred, aff_target):
		# compute regression loss
		loss_regression = (aff_pred.view(-1) - aff_target.view(-1)).abs().mean()
		if self.use_mse:
			loss_regression = self.mse(aff_pred.view(-1), aff_target.view(-1))
		loss = loss_regression
		loss_info = {
			'loss': loss_regression.item(), # this determines which checkpoint is saved
			'loss_regression': loss_regression.item(),
		}
		return loss, loss_info

	def ptg_to_dgl(self, data, distance_cutoff=5.0):
		x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
		batch = data.batch.to(self.device)
		coords = data.coords.to(self.device)
		# compute distances
		distances = self.pair_distance(coords[edge_index[0]], coords[edge_index[1]]) + 1e-4
		# get rid off unexpected edges
		edge_index = edge_index[:, distances < distance_cutoff]
		if self.atom3d:
			self.edge_attr = data.edge_attr[distances < distance_cutoff].to(self.device).view(-1, 1)
		distances = distances[distances < distance_cutoff]
		distances = distances.view(-1, 1)

		# cache variables
		self.x = x
		self.edge_index = edge_index
		self.coords = coords
		self.batch = batch
		self.distances = distances
		i = edge_index[0]
		j = edge_index[1]
		is_ligand = self.get_is_ligand(x).to(self.device)
		is_protein = is_ligand.logical_not()
		ligand_index = is_ligand.nonzero().view(-1)
		protein_index = is_protein.nonzero().view(-1)
		self.is_ligand = is_ligand
		self.is_protein = is_protein
		self.ligand_index = ligand_index.to(self.device)
		self.protein_index = protein_index.to(self.device)
		self.from_protein = (i[..., None] == protein_index).any(-1).squeeze()
		self.from_ligand = (i[..., None] == ligand_index).any(-1).squeeze()
		self.to_ligand = (j[..., None] == ligand_index).any(-1).squeeze()
		self.non_int_mask = (self.from_ligand == self.to_ligand).to(self.device)

		# make DGL graph
		src = edge_index[0]
		dst = edge_index[1]
		# Create graph
		G = dgl.graph((src, dst), num_nodes=x.shape[0])
		# Add node features to graph
		G.ndata['x'] = coords #[num_atoms,3]
		G.ndata['f'] = x.unsqueeze(-1) #[num_atoms,in_dim,1]
		# Add edge features to graph
		G.edata['d'] = coords[dst] - coords[src] #[num_atoms,3]
		if self.distance_bias:
			G.edata['distance_bias'] = self.distance_bias(distances) # -2 * distances.log()
		return G

	def reduce(self, x0, x, edge_index, edge_attr, batch, distances=None):
		if self.reduce_type == 'edge':
			if self.reduce_from == 'all':
				reduce_edge_attr = edge_attr
				reduce_batch = batch[edge_index[0]]
			elif self.reduce_from == 'ligand':
				ligand_index = (self.get_is_ligand(x0)).nonzero().view(-1)
				i = edge_index[0]
				j = edge_index[1]
				from_ligand = (i[..., None] == ligand_index).any(-1).squeeze()
				to_ligand = (j[..., None] == ligand_index).any(-1).squeeze()
				reduce_mask = (from_ligand + to_ligand) >= 1 # at least one end connects with ligand
				reduce_edge_attr = edge_attr[reduce_mask]
				reduce_batch = batch[i[reduce_mask]]
			elif self.reduce_from == 'ligand-protein':
				ligand_index = (self.get_is_ligand(x0)).nonzero().view(-1)
				i = edge_index[0]
				j = edge_index[1]
				from_ligand = (i[..., None] == ligand_index).any(-1).squeeze()
				to_ligand = (j[..., None] == ligand_index).any(-1).squeeze()
				reduce_mask = (from_ligand + to_ligand) == 1
				reduce_edge_attr = edge_attr[reduce_mask]
				reduce_batch = batch[i[reduce_mask]]
			else:
				print(f'Reducing over [{self.reduce_from} {self.reduce_type}] is undefiend')
				exit(1)
			out = self.out_mlp(0.1 * self.pool(reduce_edge_attr, reduce_batch))
		elif self.reduce_type == 'node':
			if self.reduce_from == 'all':
				scale_factor = 100
				reduce_x = x
				reduce_batch = batch
			elif self.reduce_from == 'ligand':
				scale_factor = 10
				ligand_index = (self.get_is_ligand(x0)).nonzero().view(-1)
				reduce_x = x[ligand_index]
				reduce_batch = batch[ligand_index]
			else:
				print(f'Reducing over [{self.reduce_from} {self.reduce_type}] is undefiend')
				exit(1)
			if self.extensive:
				out = self.pool(self.out_mlp(reduce_x), reduce_batch) / scale_factor
			else:
				out = self.out_mlp(self.pool(reduce_x, reduce_batch))
		else:
			print(f'Reducing over [{self.reduce_from} {self.reduce_type}] is undefiend')
			exit(1)
		return out

	def forward_and_return_loss(self, data, return_y=False):
		y_target = data['y'].float().to(self.device)
		y_pred = self(data)
		loss, loss_info = self.loss(y_pred, y_target)
		if return_y:
			return loss, loss_info, y_target , y_pred
		return loss, loss_info

	def validate_model(self, loader, prefix):
		self.eval()
		sum_loss_info = {}
		num_batch = 0
		y_target_list = []
		y_pred_list = []
		with torch.no_grad():
			for data in tqdm(loader):
				loss, loss_info, y_target, y_pred = self.forward_and_return_loss(data, return_y=True)
				y_target_list.extend(y_target.cpu().tolist())
				y_pred_list.extend(y_pred.cpu().tolist())
				sum_loss_info = append_to_dict(sum_loss_info, loss_info)
				num_batch += 1
		average_loss_info = average_dict(sum_loss_info)
		y_true = np.array(y_target_list)
		y_pred = np.array(y_pred_list)
		r2 = r2_score(y_true=y_true, y_pred=y_pred)
		mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
		mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
		pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
		spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))
		info = {
			# 'loss': mse,
			'mae': mae,
			'rmse': mse ** 0.5,
			'pearsonr': pearsonr[0],
			'spearmanr': spearmanr[0],
			'r2': r2,
			# 'y_true': y_true,
			# 'y_pred': y_pred,
		}
		info.update(average_loss_info)
		return info

	def get_is_ligand(self, x):
		if self.atom3d:
			return x[:, 18:].sum(-1) == 1
		else:
			return x[:, 14] == 1

	def test_variation(self, data, perturb, scales):
		# Given a batch of (B,) crystal structures, return the variation of energy (N,) for perturbations of a range of scales (N,)
		# perturb: scale parameter s, coords (..., 3) -> perturbed coords (..., 3)
		d = copy.deepcopy(data)
		is_ligand = self.get_is_ligand(d.x)
		energies = []
		for s in scales:
			# change ligand coordinates
			d.coords[is_ligand] = perturb(data.coords[is_ligand], s)
			# generate graphs
			G = self.ptg_to_dgl(d, distance_cutoff=self.distance_cutoff).to(self.device)
			# evaluate energies
			energies.append(self.forward_critic(G))
		energies = torch.stack(energies, dim=0) # (N, B)
		return energies

	def e3_transform(self, x, rotation, translation):
		if rotation.shape[-1] == 3:
			# 3x3 rotation matrices
			x = torch.einsum('ijk, ik->ij', rotation, x)
		elif rotation.shape[-1] == 4:
			# quaternions
			x = qrot(rotation, x)
		x = x + translation
		return x

	def random_translate(self, x, s):
		# translate each coordinate identically in a random direction for a distance of s
		u = random_unit_3vector(1)
		dx = s * u.view(-1, 3) # (1, 3)
		return x + dx

	def random_rotate(self, x, s):
		# apply a rotation to each coordinate along a random axis for an angle of s radians
		u = random_unit_3vector(1).view(-1)
		r = s * u # rotation vector, whose norm, i.e s, represents angle of rotation in radians
		R = Rotation.from_rotvec(r)
		rotation = torch.FloatTensor(R.as_matrix()) # (3, 3)
		x = torch.einsum('ij, nj->ni', rotation, x)
		return x

	def sample_uniform(self, x, batch):
		# Sample translations vector from a ball of radius=self.radius, and a random rotation
		batch_size = batch.max() + 1
		v = torch.randn(batch_size, 3) # random vector
		u = v / (v.norm(dim=-1) + 1e-4).view(-1, 1) # normalize
		s = self.radius * torch.rand(batch_size, 1) ** (1/3) # random distance
		t = s * u # random translation: (B, 3)
		rotation = torch.FloatTensor(Rotation.random(batch_size).as_matrix()) # (B, 3, 3)
		return self.e3_transform(x, rotation[batch], t[batch])

def count_parameter(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)
