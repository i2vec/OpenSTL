import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
from openstl.utils import get_dataset
from openstl.methods import method_maps
import torch
from openstl.utils.logger import SetupCallback
import sys
import datetime
import argparse

class PLBaseExperiment:
    
    def __init__(self, args, dataloaders=None):
        self.args = args
        self.config = self.args.__dict__
        self.method = None
        self.args.method = self.args.method.lower()
        self._preparation(dataloaders)
        self._init_trainer(args)
        self.data_module = DataModule(
            self.train_loader, 
            self.valid_loader, 
            self.test_loader
        )
        self.method.hparams.test_loader = self.test_loader
        
    def train(self):
        self.trainer.fit(self.method, self.data_module)
        
    def _init_trainer(self, args):
        trainer_config = {
            'local_rank': args.local_rank, 
            'port': args.port, 
            'gpus': args.gpus,  # Use the all GPUs
            'max_epochs': args.epoch,  # Maximum number of epochs to train for
            'num_nodes': args.num_nodes,  # Number of nodes to use for distributed training
            "strategy": 'ddp', # 'ddp', 'deepspeed_stage_2
            # "precision": 'bf16', # "bf16", 16
            # 'auto_scale_batch_size': 'binsearch',
            'accelerator': 'gpu',  # Use distributed data parallel
            'callbacks': self.load_callbacks(args),
            'logger': plog.WandbLogger(
                        project = args.project,
                        name=args.ex_name,
                        save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                        offline = args.offline,
                        id = args.ex_name,
                        entity = args.entity),
            'gradient_clip_val':1.0
        }
        print(trainer_config)
        trainer_opt = argparse.Namespace(**trainer_config)
        self.trainer = Trainer.from_argparse_args(trainer_opt)
        
    
    def _preparation(self, dataloaders=None):
        
        seed_everything(self.args.seed)
        self._get_data(dataloaders)
        self._build_method()
        
    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            self.train_loader, self.valid_loader, self.test_loader = \
                get_dataset(self.args.dataname, self.config)
        else:
            self.train_loader, self.valid_loader, self.test_loader = dataloaders

        if self.valid_loader is None:
            self.valid_loader = self.test_loader
            
    def _build_method(self):
        self.args.steps_per_epoch = len(self.train_loader)
        self.config['steps_per_epoch'] = len(self.train_loader)
        self.method = method_maps[self.args.method](**self.config)
        
    def load_callbacks(self, args):
        callbacks = []
        
        logdir = str(os.path.join(args.res_dir, args.ex_name))
        
        ckptdir = os.path.join(logdir, "checkpoints")
        

        metric = args.metric_for_bestckpt
        sv_filename = 'best-{epoch:02d}-{val_seq_loss:.3f}'
        callbacks.append(plc.ModelCheckpoint(
            monitor=metric,
            filename=sv_filename,
            save_top_k=15,
            mode='min',
            save_last=True,
            dirpath = ckptdir,
            verbose = True,
            every_n_epochs = args.log_step,
        ))

        
        now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
        cfgdir = os.path.join(logdir, "configs")
        callbacks.append(
            SetupCallback(
                    now = now,
                    logdir = logdir,
                    ckptdir = ckptdir,
                    cfgdir = cfgdir,
                    config = args.__dict__,
                    argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],)
        )
        
        
        if args.sched:
            callbacks.append(plc.LearningRateMonitor(
                logging_interval=None))
        return callbacks

class DataModule(pl.LightningDataModule):
    def __init__(self, train_loader, valid_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader