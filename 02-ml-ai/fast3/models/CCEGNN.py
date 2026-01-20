# Create your own model!
import torch
from torch import nn
from models.base import BaseModel
from models.egnn import E_GCL
from pdb import set_trace
from torch_geometric.nn import global_mean_pool, global_add_pool

class CEGNN(BaseModel):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            loss_fn,
            hidden_channels: int = 20,
            hidden_nf: int = 1,
            dim_feedforward: int = 10,
            dropout: float = 0.1,
            atom_3d: bool =False,
            cut_off: float = 5.,
            n_p_layers: int = 5,
            n_l_layers: int = 5,
            n_lp_layers:int = 5,
            n_int_layers:int = 5,
    ):

        super(CEGNN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            loss_fn=loss_fn
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_fn = loss_fn
        self.embedding_layer = nn.Linear(
            in_channels,
            hidden_channels
        )


        self.n_p_layers = n_p_layers
        self.n_l_layers = n_l_layers
        self.n_lp_layers = n_lp_layers
        self.hidden_nf = 20
        for i in range(0, n_p_layers):
            self.add_module("gcl_p_%d" % i, E_GCL(input_nf=hidden_channels, output_nf=self.hidden_nf, hidden_nf=self.hidden_nf, edges_in_d=0,
                                                act_fn=nn.SiLU(), residual=True, attention=True,
                                                normalize=False, tanh=True))
            self.add_module("tanh_p_%d" % i, nn.Tanh())
            self.add_module("bn_p_%d" % i, nn.BatchNorm1d(hidden_channels))
            self.add_module("tanh_p_c_%d" % i, nn.Tanh())
            self.add_module("bn_p_c_%d" % i, nn.BatchNorm1d(3))
        
        for i in range(0, n_l_layers):
            self.add_module("gcl_l_%d" % i, E_GCL(input_nf=hidden_channels, output_nf=self.hidden_nf, hidden_nf=self.hidden_nf, edges_in_d=0,
                                                act_fn=nn.SiLU(), residual=True, attention=True,
                                                normalize=False, tanh=True))
            self.add_module("tanh_l_%d" % i, nn.Tanh())
            self.add_module("bn_l_%d" % i, nn.BatchNorm1d(hidden_channels))
            self.add_module("tanh_l_c_%d" % i, nn.Tanh())
            self.add_module("bn_l_c_%d" % i, nn.BatchNorm1d(3))

        for i in range(0, n_lp_layers):
            self.add_module("gcl_lp_%d" % i, E_GCL(input_nf=hidden_channels, output_nf=self.hidden_nf, hidden_nf=self.hidden_nf, edges_in_d=0,
                                                act_fn=nn.SiLU(), residual=True, attention=True,
                                                normalize=False, tanh=True))
            self.add_module("tanh_lp_%d" % i, nn.Tanh())
            self.add_module("bn_lp_%d" % i, nn.BatchNorm1d(hidden_channels))
            self.add_module("tanh_lp_c_%d" % i, nn.Tanh())
            self.add_module("bn_lp_c_%d" % i, nn.BatchNorm1d(3))

        for i in range(0, n_int_layers):
            self.add_module("gcl_int_%d" % i, E_GCL(input_nf=hidden_channels, output_nf=self.hidden_nf, hidden_nf=self.hidden_nf, edges_in_d=0,
                                                act_fn=nn.SiLU(), residual=True, attention=True,
                                                normalize=False, tanh=True))
            self.add_module("tanh_int_%d" % i, nn.Tanh())
            self.add_module("bn_int_%d" % i, nn.BatchNorm1d(hidden_channels))
            self.add_module("tanh_int_c_%d" % i, nn.Tanh())
            self.add_module("bn_int_c_%d" % i, nn.BatchNorm1d(3))

        
        self.pair_distances = nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        self.atom3d = atom_3d

        self.pool = global_add_pool

        self.cut_off = cut_off

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_channels,1),
            nn.ReLU()
        )
        self.mlp_lp = nn.Sequential(
            nn.Linear(hidden_channels,hidden_channels),
            nn.ReLU()
        )

        self.int_egcl_mlp = nn.Sequential(
            nn.Linear(3*hidden_channels,hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )

    def _forward(self, data):
        # Reduce the number of edges
        # Get the distances
        x, edge_index, edge_attrs = data.x, data.edge_index, data.edge_attrs
        coords = data.pos

        edge_attrs = self.pair_distances(coords[edge_index[0]],
                                       coords[edge_index[1]]) + 1e-4

        edge_index = edge_index[:, edge_attrs < self.cut_off]
        
        edge_attrs = edge_attrs[edge_attrs< self.cut_off]
        
        # Get ligand-ligand, protein-protein and ligand-protein features
        l_features, p_features, lp_features = self.feature_extractor(
            x=x, 
            edge_index=edge_index, 
            edge_attrs=edge_attrs
        )
        
        # Transform all the existing features into an embedding
        h = self.embedding_layer(x)


        # Feed the transformed features to an EGNN module (ligand-ligand)
        for i in range(0, self.n_p_layers):
            if i ==0:
                hbar_p, coords_hat_p, _ = self._modules["gcl_p_%d" % i](
                    h=h, 
                    edge_index=p_features["edge_index"], 
                    coord=coords, 
                    edge_attr=p_features["edge_attrs"].unsqueeze(1)
                )
                
            else:
                hbar_p, coords_hat_p, _ = self._modules["gcl_p_%d" % i](
                    h=hbar_p, 
                    edge_index=p_features["edge_index"], 
                    coord=coords_hat_p, 
                    edge_attr=p_features["edge_attrs"].unsqueeze(1)
                )
                
            hbar_p = self._modules["tanh_p_%d" %i](hbar_p)
            hbar_p = self._modules["bn_p_%d" %i](hbar_p)
            coords_hat_p = self._modules["tanh_p_c_%d" %i](coords_hat_p)
            coords_hat_p = self._modules["bn_p_c_%d" %i](coords_hat_p)

        # Feed the transformed features to an EGNN module (protein-protein)
        for i in range(0, self.n_l_layers):
            if i ==0:
                hbar_l, coords_hat_l, _ = self._modules["gcl_l_%d" % i](
                    h=h, 
                    edge_index=l_features["edge_index"],
                    coord=coords,
                    edge_attr=l_features["edge_attrs"].unsqueeze(1)
                )
                
            else:
                hbar_l, coords_hat_l, _ = self._modules["gcl_l_%d" % i](
                    h=hbar_l, 
                    edge_index=l_features["edge_index"],
                    coord=coords_hat_l,
                    edge_attr=l_features["edge_attrs"].unsqueeze(1)
                )
                
            hbar_l = self._modules["tanh_l_%d" %i](hbar_l)
            hbar_l = self._modules["bn_l_%d" %i](hbar_l)
            coords_hat_l = self._modules["tanh_l_c_%d" %i](coords_hat_l)
            coords_hat_l = self._modules["bn_l_c_%d" %i](coords_hat_l)

        

        # Feed the transformed features to an EGNN module (ligand-protein)
        for i in range(0, self.n_lp_layers):
            if i == 0:
                hbar_lp, coords_hat_lp, _ = self._modules["gcl_lp_%d" % i](
                    h=h,
                    edge_index=lp_features["edge_index"],
                    coord=coords,
                    edge_attr=lp_features["edge_attrs"].unsqueeze(1)
                )
            else:
                hbar_lp, coords_hat_lp, _ = self._modules["gcl_lp_%d" % i](
                    h=hbar_lp, 
                    edge_index=lp_features["edge_index"],
                    coord=coords_hat_lp,
                    edge_attr=lp_features["edge_attrs"].unsqueeze(1)
                )
            hbar_lp = self._modules["tanh_lp_%d" %i](hbar_lp)
            hbar_lp = self._modules["bn_lp_%d" %i](hbar_lp)
            coords_hat_lp = self._modules["tanh_lp_c_%d" %i](coords_hat_lp)
            coords_hat_lp = self._modules["bn_lp_c_%d" %i](coords_hat_lp)

        hbar_int = torch.cat([hbar_l, hbar_p, hbar_lp], axis=1)
        hbar_int = self.int_egcl_mlp(hbar_int)
        
        # Feed the transformed features to an EGNN module (final)
        for i in range(0, self.n_lp_layers):
            if i == 0:
                hbar_int, coords_hat_int, _ = self._modules["gcl_int_%d" % i](
                    h=hbar_int,
                    edge_index=edge_index,
                    coord=coords,
                    edge_attr=edge_attrs.unsqueeze(1)
                )
                
            else:
                hbar_int, coords_hat_int, _ = self._modules["gcl_int_%d" % i](
                    h=hbar_int, 
                    edge_index=edge_index,
                    coord=coords_hat_int,
                    edge_attr=edge_attrs.unsqueeze(1)
                )
                
            hbar_int = self._modules["tanh_int_%d" %i](hbar_int)
            hbar_int = self._modules["bn_int_%d" %i](hbar_int)
            coords_hat_int = self._modules["tanh_int_c_%d" %i](coords_hat_int)
            coords_hat_int = self._modules["bn_int_c_%d" %i](coords_hat_int)

        # Apply pooling to coords_hat_l
        out = self.pool(hbar_int, data.batch)
        return self.mlp_out(out)



    def feature_extractor(self, edge_index, x, edge_attrs):
        
        # Get ligand index
        ligand_index = (x[:, 14] == 1).nonzero().view(-1)
        # Get protein index
        protein_index = (x[:, 14] == 1).logical_not().nonzero().view(-1)

        # Extract edge-ends
        edge_0, edge_1 = edge_index[0], edge_index[1]

        # Extract ligand-indices and protein-indices from edges
        from_ligand = (edge_0[..., None] ==ligand_index).any(-1).squeeze()
        to_ligand = (edge_1[..., None] == ligand_index).any(-1).squeeze()

        from_protein = (edge_0[..., None] == protein_index).any(-1).squeeze()
        to_protein = (edge_1[..., None] == protein_index).any(-1).squeeze()

        # Only keep ligand-ligand edges, protein-protein and ligand-protein edges
        ll_indices = (from_ligand.int() + to_ligand.int()) > 1
        pp_indices = (from_protein.int() + to_protein.int()) > 1
        lp_indices = (from_ligand.int() + to_protein.int()) > 1

        
        # Extract ligand edges, protein edges and ligand-protein edges
        ll_edge_index = edge_index[:, ll_indices]
        pp_edge_index = edge_index[:, pp_indices]
        lp_edge_index = edge_index[:, lp_indices]

        # Extract the attrs for l-l, l-p and p-p edges
        ll_attrs = edge_attrs[ll_indices]
        pp_attrs = edge_attrs[pp_indices]
        lp_attrs = edge_attrs[lp_indices]

        # send as dictionaries
        ligand_features = {
            "edge_index": ll_edge_index,
            "edge_attrs": ll_attrs
        }

        protein_features = {
            "edge_index": pp_edge_index,
            "edge_attrs": pp_attrs
        }

        ligand_protein_features = {
            "edge_index": lp_edge_index,
            "edge_attrs": lp_attrs,
        } 

        return (
            ligand_features,
            protein_features, 
            ligand_protein_features
        )