import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
import os
import torch.nn as nn
from openstl.utils.utils import get_text_logger
from openstl.core import get_optim_scheduler, metric


class Base_plmethod(pl.LightningModule):
    
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)
        self.text_logger = get_text_logger(self.hparams.res_dir, self.hparams.ex_name) # 你能使用text_logger在本地的.log文件中记录任何信息
    
    def _build_model(self):
        raise NotImplementedError

    def forward(self, batch):
        results = self.model(batch[0])
        return results
    

    def training_step(self, batch, batch_idx, **kwargs):
        results = self(batch)
        loss = self.loss_function(results, batch[1])
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        results = self(batch)
        
        # acc = (results.argmax(dim=-1) == batch[1]).float().mean()
        loss = self.loss_function(results, batch[1])
        
        if 'weather' in self.hparams.dataname:
            metric_list, spatial_norm = self.hparams.metrics, True
            channel_names = self.hparams.data_module.test_loader.dataset.data_name if 'mv' in self.hparams.dataname else None
        else:
            metric_list, spatial_norm, channel_names = self.hparams.metrics, False, None
        
        eval_res, _ = metric(
            results.cpu().numpy(), 
            batch[1].cpu().numpy(), 
            self.hparams.test_loader.dataset.mean, 
            self.hparams.test_loader.dataset.std,
            metrics=metric_list, 
            channel_names=channel_names, 
            spatial_norm=spatial_norm
        )

        log_metrics = {
            'val_loss': loss
        }
        for metri in metric_list:
            log_metrics[metri] = eval_res[metri]
        self.log_dict(log_metrics)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        
        optimizer, schecular, _ = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        
        return [optimizer], [{"scheduler": schecular, "interval": "step"}]
        
    def lr_scheduler_step(self, *args, **kwargs):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def configure_loss(self, loss_func):
        self.loss_function = loss_func