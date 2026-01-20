import torch
import pytorch_lightning as pl
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef
from torch.nn import Linear

class BaseModel(pl.LightningModule):
    def __init__(self, in_dim, out_dim, output_dir, optimizer=None, schedule_lr=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.output_dir = output_dir
        self.define_modules()

        # todo: remove this, can possible lead to memory leaks 
        # configure metrics using torchmetrics so everything can stay as a tensor
        self.r2_score = R2Score()
        self.mae = MeanAbsoluteError()
        self.rmse = MeanSquaredError(squared=False)
        self.pearsonr = PearsonCorrCoef()
        self.spearmanr = SpearmanCorrCoef()
        self.metrics = {
            "r2_score": self.r2_score, 
            "mae": self.mae, 
            "rmse": self.rmse, 
            "pearsonr": self.pearsonr, 
            "spearmanr": self.spearmanr
            }
        

    def configure_optimizers(self):
        #TODO: allow for different optimizers / schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.01, verbose=True)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

    def training_step(self, batch, batch_idx):

        batch.to(self.device)
        loss, loss_info, y_target, y_pred = self.forward_and_return_loss(batch, return_y=True)
        return {
            "loss": loss,
            "y_target": y_target.view(-1),
            "y_pred": y_pred.detach().view(-1),
        }

    def training_epoch_end(self, train_step_outputs):
        y_preds = [d['y_pred'] for d in train_step_outputs]
        y_targets = [d['y_target'] for d in train_step_outputs]
        y_preds = torch.cat(y_preds)
        y_targets = torch.cat(y_targets)
        
        train_loss = self.metrics['rmse'](y_preds, y_targets)
        for metric_name, metric in self.metrics.items():
            metric_name = "train_" + metric_name
            self.log(metric_name, metric(y_preds, y_targets))
        #return train_loss
                

    def validation_step(self, batch, batch_idx):
        '''
        Just return predictions and targets. We will accumulate these and compute metrics 
        in self.validation_epoch_end()
        '''
        # print("hello")
        # batch.to(self.device)
        # import ipdb 
        # ipdb.set_trace()

        loss, loss_info, y_target, y_pred = self.forward_and_return_loss(batch, return_y=True)

        return {"val_loss_step": loss}


    def validation_step_end(self, batch_parts):

        losses = batch_parts["val_loss_step"]
        return {"val_loss_mean": torch.mean(losses)}

    def validation_epoch_end(self, validation_step_outputs):

        loss = torch.cat([x['val_loss_mean'].reshape(-1,1) for x in validation_step_outputs]).mean()
        self.log("val_loss", loss)

        return {"val_loss": loss}
 
        
    def define_modules(self):
        self.out_mlp = Linear(self.in_dim, self.out_dim)

    def forward_and_return_loss(self, data, return_y=False):
        # y_target = data['y'].float().to(self.device)
        y_target = data['y'].float()
        # import ipdb
        # ipdb.set_trace()
        y_pred = self.forward(data)
        # print(f"forward_and_return_loss: y_target.shape={y_target.shape}\ty_pred.shape={y_pred.shape}")
        loss, loss_info = self.loss(y_pred, y_target)
        if return_y:
            return loss, loss_info, y_target, y_pred
        return loss, loss_info


    def test_step(self, batch, batch_idx):

        y_pred = self(batch)
        return y_pred


    def loss(self, y_pred, y_target):
            raise NotImplementedError

    def forward(self, data):
            raise NotImplementedError
