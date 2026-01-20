import torch.optim as optim
from MEAG_VAE.config import cfg


def build_optimizer(params):
    
    weight_decay = cfg.optim.weight_decay
    
    filter_fn = filter(lambda p : p.requires_grad, params)
    

    # filter selects those parameters that have requires_grad
  
    if cfg.optim.opt == 'adam': 
        optimizer = optim.Adam(filter_fn, lr=cfg.optim.lr, weight_decay=weight_decay)
    elif cfg.optim.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=cfg.optim.lr, momentum=0.95, weight_decay=weight_decay)
    elif cfg.optim.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=cfg.optim.lr, weight_decay=weight_decay)
    elif cfg.optim.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=cfg.optim.lr, weight_decay=weight_decay)
        
        
    if cfg.optim.scheduler == 'none':
        return None, optimizer
    elif cfg.optim.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.optim.opt_decay_step, gamma=cfg.optim.opt_decay_rate)
        # multiply lr by gamma every step_size epochs
        
    elif cfg.optim.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optim.opt_restart)
    return scheduler, optimizer
