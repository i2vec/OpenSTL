import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from timm.utils import AverageMeter
from openstl.models import PredNet_Model
from openstl.utils import (reduce_tensor, get_initial_states)
from .base_plmethod import Base_plmethod


class PredNet(Base_plmethod):
    r"""PredNet

    Implementation of `Deep Predictive Coding Networks for Video Prediction
    and Unsupervised Learning <https://arxiv.org/abs/1605.08104>`_.

    """
    
    def __init__(self, **config):
        super().__init__(**config)
        self.model = self._build_model(self.hparams)
        self.criterion = nn.MSELoss()
        self.configure_loss(self.criterion)


    def _build_model(self, args):
        return PredNet_Model(args.stack_sizes, args.R_stack_sizes,
                             args.A_filt_sizes, args.Ahat_filt_sizes,
                             args.R_filt_sizes, args.pixel_max, args)

    def _predict(self, batch_x, batch_y, **kwargs):
        input = torch.cat([batch_x, batch_y], dim=1)
        states = get_initial_states(input.shape, -2, -1, len(self.args.stack_sizes),
                                    self.args.R_stack_sizes, self.args.stack_sizes,
                                    -3, self.args.device)
        predict_list, _ = self.model(input, states, extrapolation=True)
        pred_y = torch.stack(predict_list[batch_x.shape[1]:], dim=1)
        return pred_y
