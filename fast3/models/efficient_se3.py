import dgl
import torch
from torch import nn
from torch.nn import Sequential, Linear, LeakyReLU, Dropout, LayerNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, GlobalAttention

from models.base import BaseModel
from pdb import set_trace
from models.se3_transformer.model.transformer import SE3Transformer
from models.se3_transformer.model.fiber import Fiber
from models.se3_transformer.utils import assign_relative_pos
from models.se3_transformer.model.layers.linear import LinearSE3
import subprocess as sp


def get_gpu_memory(self):
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(f'GPU 1: {memory_free_values[0]} GPU2: {memory_free_values[1]}')


class EfficientSE3Transformer(BaseModel):

    def __init__(self,
                 in_channels,
                 out_channels,
                 loss_fn,
                 edge_dim=4,
                 num_convs=3,
                 fiber_mid=None,
                 fiber_edge=None,
                 heads=1,
                 out_multiplicity=16,
                 dropout=0.,
                 reduce_type='edge',
                 reduce_from='ligand-protein',
                 pool_method='add',
                 uniform_attention=True,
                 pair_bias=True,
                 x_ij='add',
                 save_params=False,
                 share_filter=False,
                 distance_bias=False,
                 distance_cutoff=1e6,
                 atom3d=False):
        super().__init__(in_channels, out_channels, loss_fn)
        self.edge_dim = edge_dim
        self.num_convs = num_convs
        self.fiber_mid = fiber_mid
        self.heads = heads
        self.out_multiplicity = out_multiplicity
        self.dropout = dropout
        self.reduce_type = reduce_type
        self.reduce_from = reduce_from
        self.pool_method = pool_method
        self.uniform_attention = uniform_attention
        self.pair_bias = pair_bias
        self.x_ij = x_ij
        self.save_params = save_params
        self.share_filter = share_filter
        self.distance_bias = distance_bias
        self.distance_cutoff = distance_cutoff
        self.atom3d = atom3d
        self.fiber_edge = fiber_edge
        self.dropout = dropout
        

        self.define_modules()
        self.pair_distance = torch.nn.PairwiseDistance(p=2)
        

    def define_modules(self):
        
        # Linear(self.in_channels, 16)
        self.se3_interaction = SE3Transformer(
            channels_div=16,
            fiber_in=Fiber(structure={0: self.out_multiplicity}),
            fiber_hidden=Fiber(structure=self.fiber_mid),
            fiber_out=Fiber(structure={0: self.out_multiplicity}),
            fiber_edge = Fiber(structure=self.fiber_edge),
            num_layers=self.num_convs,
            atom_feature_size=16,
            edge_dim=self.edge_dim,
            div=2,
            out_dim=self.out_channels,
            num_heads=self.heads,
            dropout=self.dropout,
            pair_bias=self.pair_bias,
            uniform_attention=self.uniform_attention,
            save_params=self.save_params,
            share_filter=self.share_filter,
            x_ij=self.x_ij)
        if self.reduce_type == 'node':
            hidden_dim = self.se3_interaction.fibers['out'].n_features
        else:
            hidden_dim = self.edge_dim
        self.pool = global_add_pool
        if self.pool_method == 'mean':
            self.pool = global_mean_pool
        if self.pool_method == 'attention':
            self.attn_nn = Sequential(LayerNorm(hidden_dim),
                                      Dropout(self.dropout),
                                      Linear(hidden_dim, hidden_dim),
                                      LeakyReLU(), Dropout(self.dropout),
                                      Linear(hidden_dim, 1))
            self.value_nn = Sequential(LayerNorm(hidden_dim),
                                       Dropout(self.dropout),
                                       Linear(hidden_dim, hidden_dim),
                                       LeakyReLU(), Dropout(self.dropout),
                                       Linear(hidden_dim, hidden_dim))
            self.pool = GlobalAttention(self.attn_nn, self.value_nn)
        
        self.out_mlp = dgl.nn.pytorch.conv.EGNNConv(
            in_size=hidden_dim,
            hidden_size=hidden_dim,
            out_size=self.out_channels
        )
        self.interaction_linear_edge = dgl.nn.pytorch.conv.EGNNConv(
            in_size=hidden_dim,
            hidden_size=hidden_dim,
            out_size=self.out_multiplicity
        )
        self.linear_node = dgl.nn.pytorch.conv.EGNNConv(
            in_size=20,
            hidden_size=hidden_dim,
            out_size=self.out_multiplicity
        )

    def _forward(self, data, get_feature=False):
        # G = self.ptg_to_dgl(data, distance_cutoff=self.distance_cutoff)
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        coords = data.pos
        # compute distances
        distances = self.pair_distance(coords[edge_index[0]],
                                       coords[edge_index[1]]) + 1e-4
        # get rid off unexpected edges
        edge_index = edge_index[:, distances < self.distance_cutoff]
        if self.atom3d:
            edge_attr = data.edge_attr[distances < self.distance_cutoff].view(-1, 1)
        distances = distances[distances < self.distance_cutoff]
        distances = distances.view(-1, 1)

        #################### make DGL graph ####################
        src = edge_index[0]
        dst = edge_index[1]

        G = dgl.graph((src, dst), num_nodes=x.shape[0])
        # Add node features to graph
        G.ndata['x'] = coords  #[num_atoms,3]
        G.ndata['f'] = x.unsqueeze(-1)  #[num_atoms,in_channels,1]
        # Add edge features to graph
        G.edata['d'] = coords[dst] - coords[src]  #[num_atoms,3]
        if self.distance_bias:
            G.edata['distance_bias'] = self.distance_bias(
                distances)  # -2 * distances.log()
        G.edata['distances'] = distances

        # Assign relativep positions
        G = assign_relative_pos(G, coords=coords)

        G.ndata['f'], _ = self.linear_node(G, node_feat=G.ndata['f'].squeeze(-1), coord_feat=G.ndata['x'], edge_feat=distances)
        h_feat, _ = self.interaction_linear_edge(G, node_feat=G.ndata['f'].squeeze(-1), coord_feat=G.ndata['x'], edge_feat=distances)

        G.edata['w'] = h_feat[dst] - h_feat[src] 

        ########################################################
        ###################Do the forward pass##################

        h = self.se3_interaction(G, node_feats={'0': G.ndata['f'].unsqueeze(-1)}, edge_feats ={'0': G.edata['w'].unsqueeze(-1)})
        h = h['0']

        
        aff, feat = self.reduce(G, x, h, edge_index,
                                        distances, batch, distances)  # (B)
        # Return affinity
        if get_feature:
            return feat
        
        return aff
        ########################################################

    '''
    Reduction - Reduces to batch size depending on ligand-ligand, protein-protein or ligand-protein interactions
        x0: 
        h:
        self.edge_index:
        edg_feat:
        self.batch:
        self.distances:
    '''
    def reduce(self, G, x0, x, edge_index, edge_attr, batch, distances=None):
        if self.reduce_type == 'edge':
            if self.reduce_from == 'all':
                reduce_features = x
                reduce_batch = batch
            elif self.reduce_from == 'ligand':
                ligand_index = (self.get_is_ligand(x0)).nonzero().view(-1)
                i = edge_index[0]
                j = edge_index[1]
                from_ligand = (i[..., None] == ligand_index).any(-1).squeeze()
                to_ligand = (j[..., None] == ligand_index).any(-1).squeeze()
                reduce_mask = (from_ligand + to_ligand
                               ) >= 1  # at least one end connects with ligand
                # reduce_edge_attr = edge_attr[reduce_mask]
                reduce_features = x[i[reduce_mask]]
                reduce_batch = batch[i[reduce_mask]]
            elif self.reduce_from == 'ligand-protein':
                ligand_index = (self.get_is_ligand(x0)).nonzero().view(-1)
                i = edge_index[0]
                j = edge_index[1]
                from_ligand = (i[..., None] == ligand_index).any(-1).squeeze()
                to_ligand = (j[..., None] == ligand_index).any(-1).squeeze()
                reduce_mask = (from_ligand + to_ligand) == 1
                reduce_features = x[i[reduce_mask]]
                reduce_batch = batch[i[reduce_mask]]
            else:
                print(
                    f'Reducing over [{self.reduce_from} {self.reduce_type}] is undefiend'
                )
                exit(1)
            # out = self.out_mlp(0.1 * self.pool(reduce_edge_attr, reduce_batch))
            out, feat = self.out_mlp(G, node_feat=x.squeeze(-1), coord_feat=G.ndata['x'], edge_feat=edge_attr)
            out = self.pool(out, reduce_batch)
            
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
                print(
                    f'Reducing over [{self.reduce_from} {self.reduce_type}] is undefiend'
                )
                exit(1)
            if self.extensive:
                # TODO(june): How to define the reduced feature?
                feat = None
                out = self.pool(self.out_mlp(reduce_x),
                                reduce_batch) / scale_factor
            else:
                out, feat = self.out_mlp(G, node_feat=x.squeeze(-1), coord_feat=G.ndata['x'], edge_feat=edge_attr)
                out = self.pool(out, reduce_batch)
        else:
            print(
                f'Reducing over [{self.reduce_from} {self.reduce_type}] is undefiend'
            )
            exit(1)
        return out, feat

    def get_is_ligand(self, x):
        if self.atom3d:
            return x[:, 18:].sum(-1) == 1
        else:
            return x[:, 14] == 1
