import torch
import os
import logging

from MEAG_VAE.args import parse_args
from MEAG_VAE.config import load_cfg,update_out_dir,cfg,dump_cfg

from MEAG_VAE.models.model_Net import Net
from MEAG_VAE.optimizer import build_optimizer
from MEAG_VAE.loader import data_loader
from MEAG_VAE.train import train_cp

from MEAG_VAE.dataprocessing.load_data import StructureDataset


if __name__ == '__main__':
    
   # args=parse_args()
    args=parse_args()
    load_cfg(cfg,args)
    print(cfg.model.file_name)
    print(cfg.model.channels)

    
    update_out_dir(cfg.out_dir,args.cfg_file)
    dump_cfg(cfg)
   

    data_set=StructureDataset(root=cfg.dataset.dir_name,
                              element=cfg.dataset.elem,
                              feature_type=cfg.dataset.feature_type)
    
    print(f'length:{len(data_set)}')
    train_set,test_set=data_loader(data_set,cfg.train.batch_size,cfg.train.train_val_ratio)
   
    channels = list(map(int, cfg.model.channels.split(",")))
    
    input_size=data_set.num_features
    print(f'num_features:{input_size},channels:{channels}')
    model = Net(input_size, cfg.model.num_kernels,cfg.model.pooling_rate, channels,cfg.model.edge_reduction)

    scheduler,opt=build_optimizer(model.parameters())
    torch.cuda.empty_cache()
    _,_,min_loss,best_model=train_cp(model, opt, scheduler, cfg.device, train_set, test_set, cfg.train.epochs,cfg.model.fixed_rate_l,cfg.model.fixed_rate_r)
   
  #  if not os.path.exists(model_savedir):
  #      os.makedirs(model_savedir)
   # torch.save(best_model,f'{cfg.out_dir}/{cfg.model.file_name}.pt')
    torch.save(best_model.state_dict(),f'{cfg.out_dir}/{cfg.model.file_name}.pth')
    print("model saved")
    print(f"min_loss:{min_loss}")
    logging.info(model)
    logging.info(cfg)









