from torch import nn
import torch
from models.base import BaseModel
from torch_geometric.nn import global_mean_pool, global_add_pool
from pdb import set_trace

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(
            self, 
            input_nf, 
            output_nf, 
            hidden_nf, 
            edges_in_d=0, 
            act_fn=nn.SiLU(), 
            residual=True, 
            attention=False, 
            normalize=False, 
            coords_agg='mean', 
            tanh=False
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):

        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr



class EGNN(BaseModel):
    '''
        The original implementation of this code can be found here : https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py
        The paper linked to the above can be found here: https://arxiv.org/pdf/2102.09844
    '''
    def __init__(
                self, 
                in_channels: int,
                out_channels: int,
                loss_fn,
                label_channels=2,
                in_edge_nf=1,
                distance_cutoff=1.5,
                act_fn=nn.SiLU(), 
                n_layers=4,
                pose=False,
                residual=True, 
                attention=True, 
                normalize=False, 
                tanh=False,
                freeze=False,
                add_layer=False,
                estlabel=False,
                softmax=False,
            ):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''
        super(EGNN, self).__init__(in_channels=in_channels, out_channels=out_channels, loss_fn=loss_fn, estlabel=estlabel)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_fn = loss_fn
        self.hidden_channels = 20
        self.hidden_nf = 20
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(self.in_channels, self.hidden_channels)
        self.embedding_out = nn.Linear(self.hidden_nf, out_channels)
        self.label_channels = out_channels
        if estlabel:
            self.label_out = nn.Sequential(
                nn.Linear(self.hidden_nf, label_channels)
            )
        self.estLabel = estlabel
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = global_add_pool
        self.labelpool = global_mean_pool
        self.relu = nn.ReLU()
        self.pose = pose
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_channels, self.hidden_nf, self.hidden_nf, edges_in_d=0,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
            self.add_module("tanh_%d" % i, nn.ReLU())
            self.add_module("tanh_c_%d" % i, nn.ReLU())
            self.add_module("bn_%d" % i, nn.BatchNorm1d(self.hidden_nf))

        
        self.pair_distance = nn.PairwiseDistance(p=2)
        self.distance_cutoff = distance_cutoff
        self.n_layers = n_layers
        self.freeze = freeze
        
        self.softmax = softmax
        if self.softmax:
            self.softmax = nn.Softmax(dim = 1)

        self.add_layer = add_layer

        self._initialize()

    
    def _initialize(self):
        if self.freeze:
            for i in range(0, self.n_layers):
                self._modules["gcl_%d" % i].requires_grad=False
                self._modules["tanh_%d" %i].requires_grad=False
                self._modules["bn_%d" %i].requires_grad=False
                self._modules["tanh_c_%d"%i].requires_grad=False
        
        if self.add_layer:
            self.add_module("lin_2", nn.Linear(
                self.hidden_nf, self.label_channels
            ))
    
    def __str__(self):
        pass

    def _forward(self, data, get_feature=False):
        x, edge_index = data.x, data.edge_index
        out = self.embedding_in(x)
        coords = data.pos
        # compute distances
        distances = self.pair_distance(coords[edge_index[0]],
                                       coords[edge_index[1]]) + 1e-4
        edge_index = edge_index[:, distances < self.distance_cutoff]
        edge_attr = data.edge_attr[distances < self.distance_cutoff].view(-1,1)
        for i in range(0, self.n_layers):
            out, coords, _ = self._modules["gcl_%d" % i](out, edge_index, coords, edge_attr=edge_attr)
            out = self._modules["tanh_%d" %i](out)
            out = self._modules["bn_%d" %i](out)
            coords = self._modules["tanh_c_%d"%i](coords)

        if self.estLabel:
            label = self.label_out(out)
        
        out = self.embedding_out(out)
        
        out = self.pool(out, data.batch)
        if get_feature:
            raise NotImplementedError("Currently does not support fusion")
        
        if not self.estLabel:
            return out # get the label too
        else:
            if self.softmax:
                return out, self.softmax(self.labelpool(label, data.batch))
            else:
                return out, self.labelpool(label, data.batch)

class EGNNmod(BaseModel):
    def __init__(
                self, 
                in_channels: int,
                out_channels: int,
                loss_fn,
                in_edge_nf=1,
                distance_cutoff=1.5,
                act_fn=nn.SiLU(), 
                n_layers=4,
                pose=False,
                residual=True, 
                attention=True, 
                normalize=False, 
                tanh=False,
                freeze=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNNmod, self).__init__(in_channels=in_channels, out_channels=out_channels, loss_fn=loss_fn)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_fn = loss_fn
        self.hidden_channels = 20
        self.hidden_nf = 20
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(self.in_channels, self.hidden_channels)
        self.embedding_out = nn.Linear(self.hidden_nf, out_channels)
        self.embedding_features = nn.Linear(self.hidden_channels, 16)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = global_mean_pool
        self.relu = nn.ReLU()
        self.pose = pose
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_channels, self.hidden_nf, self.hidden_nf, edges_in_d=0,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
            self.add_module("tanh_%d" % i, nn.Tanh())
            self.add_module("tanh_c_%d" % i, nn.Tanh())
            self.add_module("bn_%d" % i, nn.BatchNorm1d(self.hidden_nf))
            self.add_module("bn_c_%d" % i, nn.BatchNorm1d(3))

        self.pair_distance = nn.PairwiseDistance(p=2)
        self.distance_cutoff = distance_cutoff
        self.n_layers = n_layers
        self.freeze = freeze
        self._initialize()
    
    def _initialize(self):
        if self.freeze:
            for i in range(0, self.n_layers):
                self._modules["gcl_%d" % i].requires_grad=False
                self._modules["tanh_%d" %i].requires_grad=False
                self._modules["bn_%d" %i].requires_grad=False
                self._modules["tanh_c_%d"%i].requires_grad=False
                self._modules["bn_c_%d" %i].requires_grad=False
            

    def _forward(self, data, get_feature=False, ligand=True, returnCoords=True):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        x = data.x
        out = self.embedding_in(x)
        coords = data.pos
        for i in range(0, self.n_layers):
            out, coords, _ = self._modules["gcl_%d" % i](out, edge_index, coords, edge_attr=edge_attr)
            out = self._modules["tanh_%d" %i](out)
            out = self._modules["bn_%d" %i](out)
            coords = self._modules["tanh_c_%d"%i](coords)
            coords = self._modules["bn_c_%d" %i](coords)
        

        features = self.embedding_features(out)
        out = self.embedding_out(out)

        out = self.pool(out, data.batch)
        if returnCoords:
            if not get_feature:
                return self.relu(out), coords
            else:
                return self.pool(features, data.batch), coords
        
        if not get_feature:
            return out
        else:
            return self.pool(features, data.batch)
        
# EGNN as a non-BaseModel entity
class EGNNmodule(nn.Module):
    def __init__(
                self, 
                in_channels: int,
                out_channels: int,
                in_edge_nf=1,
                distance_cutoff=1.5,
                act_fn=nn.SiLU(), 
                n_layers=4,
                pose=False,
                residual=True, 
                attention=True, 
                normalize=False, 
                tanh=False,
                freeze=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNNmodule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = 20
        self.hidden_nf = 20
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(self.in_channels, self.hidden_channels)
        self.embedding_out = nn.Linear(self.hidden_nf, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = global_add_pool
        self.relu = nn.ReLU()
        self.pose = pose
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_channels, self.hidden_nf, self.hidden_nf, edges_in_d=0,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
            self.add_module("tanh_%d" % i, nn.Tanh())
            self.add_module("tanh_c_%d" % i, nn.Tanh())
            self.add_module("bn_%d" % i, nn.BatchNorm1d(self.hidden_nf))

        self.pair_distance = nn.PairwiseDistance(p=2)
        self.distance_cutoff = distance_cutoff
        self.n_layers = n_layers
        self.freeze = freeze
        self._initialize()
    
    def _initialize(self):
        if self.freeze:
            for i in range(0, self.n_layers):
                self._modules["gcl_%d" % i].requires_grad=False
                self._modules["tanh_%d" %i].requires_grad=False
                self._modules["bn_%d" %i].requires_grad=False
                self._modules["tanh_c_%d"%i].requires_grad=False
            

    def forward(self, data, get_feature=False, ligand=True, returnCoords=True):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        x = data.x
        out = self.embedding_in(x)
        coords = data.pos
        for i in range(0, self.n_layers):
            out, coords, _ = self._modules["gcl_%d" % i](out, edge_index, coords, edge_attr=edge_attr)
            out = self._modules["tanh_%d" %i](out)
            out = self._modules["bn_%d" %i](out)
            coords = self._modules["tanh_c_%d"%i](coords)
        

        # features = self.embedding_features(out)
        out = self.embedding_out(out)

        # out = self.pool(out, data.batch)
        return self.relu(out)
        


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)