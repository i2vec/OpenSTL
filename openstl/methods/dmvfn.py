import time
import torch
from tqdm import tqdm
import numpy as np
from timm.utils import AverageMeter

from openstl.core import metric
from openstl.models import DMVFN_Model
from openstl.utils import reduce_tensor, LapLoss, VGGPerceptualLoss, ProgressBar, gather_tensors_batch
from .base_plmethod import Base_plmethod


class DMVFN(Base_plmethod):
    r"""DMVFN

    Implementation of `DMVFN: A Dynamic Multi-Scale Voxel Flow Network for Video Prediction
    Predictive Learning <https://arxiv.org/abs/2303.09875>`_.

    """
    
    def __init__(self, **args):
        super().__init__(**args)
        self.model = self._build_model(self.hparams)
        max_levels = 5 if args['in_shape'][-1] > 32 else 3
        self.lap = LapLoss(max_levels=max_levels, channels=args['in_shape'][1])
        self.vggloss = VGGPerceptualLoss()
        self.configure_loss(self.criterion)
        

    def _build_model(self, args):
        in_planes = args['in_planes']
        num_features = args['num_features']
        return DMVFN_Model(in_planes, num_features, args)

    def _predict(self, batch_x, batch_y):
        """Forward the model"""
        merged = self.model(torch.cat([batch_x, batch_y],
                    dim=1), training=self.args.training)
        pred_y = merged[-1]
        batch_y = batch_y.squeeze(1) # (B, C, H, W)
        loss_l1, loss_vgg = 0, 0
        for i in range(self.args.num_block):
            loss_l1 += (self.lap(merged[i], batch_y)).mean() \
                       * (self.args.gamma ** (self.args.num_block - i - 1))
        if self.args.in_shape[1] in [1, 3]:
            loss_vgg = (self.vggloss(pred_y, batch_y)).mean()
        else:
            loss_vgg = (self.vggloss(pred_y.mean(dim=1, keepdim=True),
                                     batch_y.mean(dim=1, keepdim=True))).mean()
        loss_G = loss_l1 + loss_vgg * self.args.coef
        return pred_y, loss_G

   