import time
import torch
import torch.nn as nn
import numpy as np
from timm.utils import AverageMeter
from tqdm import tqdm

from openstl.models import PhyDNet_Model
from openstl.utils import reduce_tensor
from .base_plmethod import Base_plmethod


class PhyDNet(Base_plmethod):
    r"""PhyDNet

    Implementation of `Disentangling Physical Dynamics from Unknown Factors for
    Unsupervised Video Prediction <https://arxiv.org/abs/2003.01460>`_.

    """

    def __init__(self, **config):
        super().__init__(**config)
        self.model = self._build_model(self.hparams)
        self.criterion = nn.MSELoss()
        self.configure_loss(self.criterion)
        self.constraints = self._get_constraints()
        
    def _build_model(self, args):
        return PhyDNet_Model(args)

    def _get_constraints(self):
        constraints = torch.zeros((49, 7, 7))
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind,i,j] = 1
                ind +=1
        return constraints 

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        if not self.dist:
            pred_y, _ = self.model.inference(batch_x, batch_y, self.constraints, return_loss=False)
        else:
            pred_y, _ = self.model.module.inference(batch_x, batch_y, self.constraints, return_loss=False)
        return pred_y
