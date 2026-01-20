import torch
import torch
import os
from models import egnn
from builder import build_model, build_dataloader, build_optimizer, build_scheduler
from utils import config_util
from checkpointer import Checkpointer
from builder import build_model
from atomsci.ddm.pipeline import chem_diversity as cd
from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles
import pandas as pd
# from models import


# directories = {
#     'train':'/p/vast1/bcwc/PDBBind/pdbbind2020_general/pdb/',
#     'val':'/p/vast1/bcwc/PDBBind/pdbbind2020_refined/refined-set/',
#     'test':'/p/vast1/bcwc/PDBBind/pdbbind_casf2016/rec/'
# }

# def collect_proteins():
#     for key, value in directories.items():
#         files = os.listdir(value)
#         for file in files:
#             if file.endswith(('', '', '')):
#                 pass


# # And let's load two models
# configs = config_util.load_configs('configs/egnn_denv2_finetune.yml')


# model = build_model(
#     configs["model"],
#     configs["data"]
# )

# mode = 'test'
# default_name = 'model'

# module = model


# checkpointables = {}
# optimizer = build_optimizer(
#                 configs["train"]["optimizer"], module
#             )
# scheduler = build_scheduler(
#     configs["train"]["scheduler"], 
#     optimizer
# )

# checkpointer = Checkpointer(
#     mode, model, '/p/lustre1/ranganath2/fast.tmp/test/', default_name, **checkpointables
# )

# checkpointables = {
#     "optimizer": optimizer,
#     "scheduler": scheduler,
# }

# checkpoint = checkpointer.resume_or_load(path=configs["model"]["ckpt_path"], resume=True)



# denv2 = build_dataloader(
#     configs=configs
# )


# dengue_configs = config_util.load_configs("configs/egcnn_2020.yaml")
# pdbbind = build_dataloader(
#     configs=dengue_configs
# )


# pdbbind_dataset = pdbbind.get('test')
# denv2_dataset = denv2.get('test')

# denv2_batch = next(iter(denv2_dataset))
# pdb_batch = next(iter(pdbbind_dataset))
# denv2_batch = [point['data'] for point in denv2_batch]
# pdb_batch = [point['data'] for point in pdb_batch]
# loss1 = model(denv2_batch)
# loss2 = model(pdb_batch)

# # Find similarity quotient
# def similarity_q():
#     # Take the similarity quotients of the proteins
#     ## Do this by the sequence approach (esm3)
#     ## 
#     # Take the similarity quotients of the lingands
#     # Do this by the AMPL tanimoto calculator

#     pass
    



# print(loss1)
# model.zero_grad()
# loss1['mse'].backward()
# print(flat_grad)

# print(loss2)
# model.zero_grad()
# loss2['mse'].backward()
# print(flat_grad)
# print(torch.norm(flat_grad))


class evaluator(object):
    def __init__(
            self,
            configs
    ):
        configs = config_util.load_configs(
            configs
        )
        self.model = build_model(
            configs['model'],
            configs['data']
        )
        self.configs = configs
        
        # Load checkpoint for model
        optimizer = build_optimizer(
            configs["train"]["optimizer"], 
            self.model

        )
        scheduler = build_scheduler(
            configs["train"]["scheduler"], 
            optimizer
        )
        
        checkpointables = {
            "optimizer": optimizer,
            "scheduler": scheduler,
        }

        self.checkpointer = Checkpointer(
            configs['data']['mode'], 
            self.model, 
            '/p/lustre1/ranganath2/fast.tmp/test/',
            'model', 
            **checkpointables
        )

        checkpoint = self.checkpointer.resume_or_load(
            path=self.configs['model']['ckpt_path'], 
            resume=False
        )
        # Load all dataloaders
        self.dataloaders = []
        for dataloader in configs['datas']:
            self.dataloaders.append(
                build_dataloader(
                    dataloader
                )
            )


    
    def _gather_flat_grad(self):
        views = []
        for p in self.model.parameters():
            p.retain_grad()
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, axis=0)
    
    def grad_extract(self):
        # Use this function to extract the gradients from the model
        
        # For each dataset do
        
            
        # Do a forward pass
        # collect the loss
        # Do backward
        # use _gather_flat_grad to collect the gradients for each dataset


        pass

    def extract_tanimoto(self):
        f='/p/vast1/bcwc/PDBBind/pdbbind_casf2016_str.csv'
        df=pd.read_csv(f)
        df=df.dropna(axis=0,subset=['smiles'])
        df['base_rdkit_smiles'] = base_smiles_from_smiles(df.smiles.tolist(), workers=8)
        scol='base_rdkit_smiles'
        df1=df.dropna(axis=0,subset=[scol])
        smiles_lst1=df1[scol].tolist()
        print(df1.shape,df.shape)
        hst=cd.calc_dist_smiles('ECFP','tanimoto',smiles_lst1,None)
        return hst

    def extract_esm3(self):
        # Use this function to extract the esm3 embeddings
        # Follow the same procedure as extract_tanimoto
        pass


if __name__=='__main__':
    eval = evaluator(
        configs='configs/egcnn_2020.yaml'
    )
    # pass