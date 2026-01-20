import json
import numpy as np
import os
import time
import torch
import torch_geometric
from collections import OrderedDict
import torch.distributed as dist
from torch_geometric.data import Data
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
# from fvcore.common.checkpoint import Checkpointer

from builder import (build_dataloader, build_eval_meter, build_model,
                     build_optimizer, build_scheduler)
from checkpointer import Checkpointer
from utils.general_util import log, mkdir_p
from pdb import set_trace
import subprocess as sp

class Engine(object):
    '''
        TODO: 
            Fix Raytune to run automatically
    '''
    def __init__(self, mode, configs, save_dir, load_dir="", tune=False, resume=False):
        self.mode = mode
        self.configs = configs
        self.save_dir = save_dir

        self.resume = resume
        # self.writer = SummaryWriter()

        # Determine which device to use
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if device == "cpu":
            log.warn("GPU is not available.")
        else:
            log.warn("Total of {} GPU(s) is/are available.".format(torch.cuda.device_count()))

        self.dataloaders = build_dataloader(configs, mode=mode)

        
        self.model = build_model(configs["model"], configs["data"])

        if torch.cuda.device_count() > 0:
            
            self.model = torch_geometric.nn.DataParallel(self.model) # .float()
            self.module = self.model.module
        else:
            
            self.model = torch_geometric.nn.DataParallel(self.model)
            self.module = self.model
            
        log.infov("Loading {} parameters".format(sum([parameter.numel() for parameter in self.model.parameters()])))
        self.model.to(self.device)

        # Build an optimizer and loss
        checkpointables = {}
        if mode == "train":
            self.optimizer = build_optimizer(
                self.configs["train"]["optimizer"], self.module
            )
            self.scheduler = build_scheduler(
                self.configs["train"]["scheduler"], self.optimizer
            )
            checkpointables = {
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
            }
        else:
            self.prev_grad = None
            self.optimizer = build_optimizer(
                self.configs["train"]["optimizer"], self.module
            )
        self.eval_meter = build_eval_meter(self.configs["test"])

        # Build a checkpointer
        default_name = "model"
        if mode == "test":
            if ("checkpoint" in configs["test"]) and (configs["test"]["checkpoint"] == "best"):
                default_name = "best_model"
        self.checkpointer = Checkpointer(
            mode, self.model, save_dir, default_name, **checkpointables
        )

        # Specify output directory
        if (mode == "test") and ("test" in configs["data"]["split"]):
            test_names = sorted(set([
               name+ "_" + split for name, split in configs["data"]["split"]["test"]
            ]))
            test_name = "+".join(test_names)
            self.output_dir = mkdir_p(os.path.join(save_dir, test_name))

        else:
            self.output_dir = save_dir
            # shutil.copy(self.model.__file__, self.output_dir+"model.py")

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        print(f'GPU 1: {memory_free_values[0]} GPU2: {memory_free_values[1]}')


    def to_device(self, items):
        if isinstance(items, (dict, Data)):
            for k, v in items.items():
                
                if isinstance(v, (str, int)):
                    pass 
                else:
                    items[k] = v.to(self.device)

        elif isinstance(items, tuple):
            items = tuple(x.to(self.device) for x in items)
        
        elif isinstance(items, (str,int)):
            pass
        else:
            items = items.to(self.device)
        return items


    def print_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                log.info("  {:60}: {}".format(name, param.shape))
            else:
                log.infov("  {:60}: {}".format(name, param.shape))


    def train(self, validate=True):

        ckpt_path = None
        # Initialize with a pretrained model.
        if not self.resume:
            if "ckpt_path" in self.configs["model"]:
                ckpt_path = self.configs["model"]["ckpt_path"]
            else:
                if self.configs["model"]["name"] == "fusion":
                    first_configs = self.configs["model"]["first"]
                    second_configs = self.configs["model"]["second"]
                    ckpt_path = {
                        "first": first_configs["ckpt_path"] if "ckpt_path" in first_configs else "",
                        "second": second_configs["ckpt_path"] if "ckpt_path" in first_configs else "",
                    }
            
        checkpoint = self.checkpointer.resume_or_load(path=ckpt_path, resume=self.resume)

        start_epoch, self.best_epoch = 0, 0
        self.best_metric = self.eval_meter.initial_metric
        if self.resume:
            if "epoch" in checkpoint:
                start_epoch = int(checkpoint["epoch"])
            if "best_epoch" in checkpoint:
                self.best_epoch = int(checkpoint["best_epoch"])
            if "best_metric" in checkpoint:
                self.best_metric = checkpoint["best_metric"]

        num_epochs = self.configs["train"].get("epochs", 100)
        step, next_checkpoint_step = 0, self.configs["train"].get("checkpoint_step", 100)

        self.module.initialize()
        self.print_params()
        log.info(
            "Train for {} epochs starting from epoch {}".format(num_epochs, start_epoch))

        for epoch in range(start_epoch, num_epochs):
            train_start = time.time()
            train_loss, step, next_checkpoint_step =\
                self._train_one_epoch(epoch, step, next_checkpoint_step)
            train_time = time.time() - train_start

            log.infov("[Epoch {:03d}] Training completed in {:.2f} sec".format(epoch, train_time))
            log.warn("[Epoch {:03d}] (Overall) Loss={:.4f}".format(epoch, train_loss))

            if (self.scheduler is not None) and self.scheduler.by_epoch:
                self.scheduler.step()

            if validate:
                val_start = time.time()
                self.eval_meter.reset()
                self.validate(epoch)
                val_time = time.time() - val_start

                log.infov("[Epoch {:03d}] Validation completed in {:.2f} sec".format(epoch, val_time))

                results, eval_strs = {}, []
                metric_vals = self.eval_meter.compute()
                for k, v in metric_vals.items():
                    eval_strs.append(" {}: {:.4f} ".format(str(k), v))
                    if not isinstance(v, float):
                        v = v.astype(np.float64)
                    results[k] = v
                log_msg = "[Epoch {:03d}] (Overall)".format(epoch) + "|".join(eval_strs)
                log.warn(log_msg)

                with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                    f.write(log_msg + "\n")

                if self.eval_meter.is_better(results, self.best_metric):
                    log.warn("            Break the best {}: {:.4f} -> {:.4f}".format(
                        self.eval_meter.main_metric,
                        self.best_metric,
                        results[self.eval_meter.main_metric]
                    ))
                    # self.writer.add_scalar('Loss/Best MSE (eval)', results[self.eval_meter.main_metric], epoch)
                    self.best_epoch = epoch
                    self.best_metric = results[self.eval_meter.main_metric]
                    model_name = "best_model"
                    self.checkpointer.save(
                        name=model_name,
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        best_epoch=self.best_epoch,
                        best_metric=self.best_metric,
                        step=step,
                        epoch=epoch
                    )
                    results["epoch"] = epoch
                    with open(os.path.join(self.output_dir, "valid_results.json"), "w") as f:
                        json.dump(results, f, indent=4)

        end_msg = "Training ends! Best {} (Epoch {:03d}) = {:.4f}".format(
            self.eval_meter.main_metric, self.best_epoch, self.best_metric
        )
        log.warn(end_msg)
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(end_msg)
        log.info("Output directory: {}".format(self.output_dir))


    def _train_one_epoch(self, epoch, step, next_checkpoint_step):
        dataloader = self.dataloaders.get("train")
        num_batches = len(dataloader)
        losses = []

        self.model.train()
        

        for i, batch in enumerate(dataloader):
            inputs = [self.to_device(x["data"]) for x in batch]
            loss_dict = self.model(inputs)
            loss = sum(loss_dict.values()).mean()
            loss_val = loss.cpu().data.item()
            if loss.isnan():
                print('NaN encountered')
                exit(0)
            losses.append(loss_val)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            
            
            if (self.scheduler is not None) and (not self.scheduler.by_epoch):
                self.scheduler.step()

            # Loss
            loss_str = "[Epoch {:03d} | LR={:.7f}] ({}/{}) Loss={:.4f} | ".format(
                epoch, self.scheduler.get_last_lr()[0], i, num_batches - 1, loss_val)
            loss_str += " | ".join(["{:s}={:.4f}".format(
                _type, _value.mean().item()) for _type, _value in loss_dict.items()])
            log.info(loss_str)

            # Save checkpoint
            step += 1
            if step >= next_checkpoint_step:
                self.checkpointer.save(
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    loss=loss,
                    step=step,
                    epoch=epoch,
                    best_epoch=self.best_epoch,
                    best_metric=self.best_metric,
                )
                next_checkpoint_step += self.configs["train"].get("checkpoint_step", 100)
            
            
        torch.cuda.empty_cache()
            
        return np.mean(losses), step, next_checkpoint_step


    def validate(self, epoch):
        dataloader = self.dataloaders.get("val")
        num_batches = len(dataloader)
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs = [self.to_device(x["data"]) for x in batch]
                labels = torch.cat([x.y for x in inputs])
                
                if self.configs['model']['params']['estlabel']:
                    true_labels = torch.cat([x.label for x in inputs])
                    results, predicted_labels = self.model(inputs)
                    metric_vals = self.eval_meter.update(results, labels, predicted_labels=predicted_labels, true_labels=true_labels)
                else:
                    results = self.model(inputs)
                    metric_vals = self.eval_meter.update(results, labels)
                

                batch_str = "[Epoch {:03d}] ({}/{})".format(epoch, i, num_batches - 1)
                eval_strs = []
                for k, v in metric_vals.items():
                    eval_strs.append(" {}: {:.4f} ".format(str(k), v))
                log.info(batch_str + "|".join(eval_strs))

                self.optimizer.zero_grad()

        torch.cuda.empty_cache()
        return results


    def evaluate(self):
        checkpoint = self.checkpointer.resume_or_load(resume=False)
        
        dataloader = self.dataloaders.get("test")
        num_batches = len(dataloader)
        self.model.eval()

        predictions = {}
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs = [self.to_device(x["data"]) for x in batch]
                labels = torch.cat([x.y for x in inputs])

                if self.configs['model']["params"]['estlabel']:
                    true_labels = torch.cat([x.label for x in inputs])
                    results, predicted_labels  = self.model(inputs)
                    if self.configs["data"]["split"]["test"][0][0] in ("denv2", "zika", "westnile"):
                        # set_trace()
                        predictions.update({
                            b["id"] + "_" + str(b["pose"]): 
                                {'affn':(float(r), float(b["data"].y)),
                                 'class': (l_p, l_t)} # (dt, gt)
                            for b, r, l_p, l_t in zip(
                                batch, 
                                results.cpu().numpy().reshape(-1).tolist(), 
                                torch.argmax(predicted_labels, dim=1).cpu().numpy().tolist(), 
                                torch.argmax(true_labels, dim=1).cpu().numpy().tolist()
                            )
                        })
                    else:
                        predictions.update({
                            b["id"] + "_" + str(b["pose"]): 
                                {'affn':(float(r), float(b["data"].y)),
                                 'class': (l_p, l_t)} # (dt, gt)
                            for b, r, l_p, l_t in zip(
                                batch, 
                                results.cpu().numpy().reshape(-1).tolist(), 
                                torch.argmax(predicted_labels, dim=1).cpu().numpy().tolist(), 
                                torch.argmax(true_labels.view_as(predicted_labels), dim=1).cpu().numpy().tolist()
                            )
                        })
                    metric_vals = self.eval_meter.update(results, labels, predicted_labels=predicted_labels, true_labels=true_labels)
                else:
                    results = self.model(inputs)
                    if self.configs["data"]["split"]["test"][0][0] in ("denv2", "zika", "westnile"):

                        predictions.update({
                            b["id"] + "_" + str(b["pose"]): (float(r), float(b["data"].y)) # (dt, gt)
                            for b, r in zip(batch, results)
                        })
                    else:
                        predictions.update({
                            b["id"]: (float(r), float(b["data"].y)) # (dt, gt)
                            for b, r in zip(batch, results)
                        })
                    metric_vals = self.eval_meter.update(results, labels)



                # Compute evaluation metrics
                batch_str = "[Epoch {:03d}] ({}/{})".format(checkpoint["epoch"], i, num_batches - 1)
                eval_strs = []
                for k, v in metric_vals.items():
                    eval_strs.append(" {}: {:.4f} ".format(str(k), v))
                log.info(batch_str + "|".join(eval_strs))

        results, eval_strs = {}, []
        metric_vals = self.eval_meter.compute()
        for k, v in metric_vals.items():
            eval_strs.append(" {}: {:.4f} ".format(str(k), v))
            if not isinstance(v, float):
                v = v.astype(np.float64)
            results[k] = v
        log_msg = "[Epoch {:03d}] (Overall)".format(checkpoint["epoch"]) + "|".join(eval_strs)
        log.warn(log_msg)

        results["epoch"] = checkpoint["epoch"]
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        with open(os.path.join(self.output_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=4)
    
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


    def grad_eval(self):
        checkpoint = self.checkpointer.resume_or_load(resume=False)
        dataloader = self.dataloaders.get("test")
        num_batches = len(dataloader)
        self.model.train()
        predictions = {}
        

        for i, batch in enumerate(dataloader):
            inputs = [self.to_device(x["data"]) for x in batch]

            if 'estlabel' in self.configs['model']["params"]:
                true_labels = torch.cat([x.label for x in inputs])
                results, predicted_labels = self.model(inputs)
                loss_dict = self.model(inputs)
                
                # Collect gradients
                grad = self._gather_flat_grad()
                cosine = grad.dot(grad)/grad.norm()**2
                
                if self.configs["data"]["split"]["test"][0][0] in ("denv2", "zika", "westnile"):
                    predictions.update({
                        b["id"] + "_" + str(b["pose"]): 
                            {'affn':(float(r), float(b["data"].y)),
                                'class': (l_p, l_t)} # (dt, gt)
                        for b, r, l_p, l_t in zip(
                            batch, 
                            results.cpu().numpy().reshape(-1).tolist(), 
                            torch.argmax(predicted_labels, dim=1).cpu().numpy().tolist(), 
                            torch.argmax(true_labels, dim=1).cpu().numpy().tolist()
                        )
                    })
                else:
                    predictions.update({
                        b["id"]: 
                        {
                            "affinity":(float(r), float(b["data"].y)),
                            "grad": cosine 
                        } # (dt, gt)
                        for b, r in zip(batch, results)
                    })
                
            else:
                results = self.model(inputs)
                loss_dict = self.model(inputs)
                loss = sum(loss_dict.values()).mean()
                self.optimizer.zero_grad()
                loss.backward()
                grad = self._gather_flat_grad()
                if self.prev_grad is None:
                    self.prev_grad = grad
                
                grad = self._gather_flat_grad()
                cosine = grad.dot(self.prev_grad)/(self.prev_grad.norm()*grad.norm())

                self.prev_grad = grad


                if self.configs["data"]["split"]["test"][0][0] in ("denv2", "zika", "westnile"):
                    predictions.update({
                        batch[0]["id"] + "_" + str(batch[0]["pose"]): grad.tolist() # (dt, gt)
                    })
                else:
                    predictions.update({
                        batch[0]["id"]: grad.tolist()
                    })

            # Compute evaluation metrics
            batch_str = "[Epoch {:03d}] ({}/{})".format(checkpoint["epoch"], i, num_batches - 1)
            eval_strs = []
            eval_strs.append(" cosine: {:.4f} ".format(cosine))
            log.info(batch_str + "|".join(eval_strs))

        results["epoch"] = checkpoint["epoch"]

        with open(os.path.join(self.output_dir, "cosine.json"), "w") as f:
            json.dump(predictions, f, indent=4)

