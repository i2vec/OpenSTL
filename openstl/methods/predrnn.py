import time
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm

from openstl.models import PredRNN_Model
from openstl.utils import (reduce_tensor, reshape_patch, reshape_patch_back,
                           reserve_schedule_sampling_exp, schedule_sampling)
from .base_plmethod import Base_plmethod


class PredRNN(Base_plmethod):
    r"""PredRNN

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://dl.acm.org/doi/abs/10.5555/3294771.3294855>`_.

    """
    def __init__(self, **config):
        super().__init__(**config)
        self.model = self._build_model(self.hparams)
        self.criterion = nn.MSELoss()
        self.configure_loss(self.criterion)
        

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PredRNN_Model(num_layers, num_hidden, args).to(self.device)

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        # reverse schedule sampling
        if self.args.reverse_scheduled_sampling == 1:
            mask_input = 1
        else:
            mask_input = self.args.pre_seq_length
        _, img_channel, img_height, img_width = self.args.in_shape

        # preprocess
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        test_dat = reshape_patch(test_ims, self.args.patch_size)
        test_ims = test_ims[:, :, :, :, :img_channel]

        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            self.args.total_length - mask_input - 1,
            img_height // self.args.patch_size,
            img_width // self.args.patch_size,
            self.args.patch_size ** 2 * img_channel)).to(self.device)
            
        if self.args.reverse_scheduled_sampling == 1:
            real_input_flag[:, :self.args.pre_seq_length - 1, :, :] = 1.0

        img_gen, _ = self.model(test_dat, real_input_flag, return_loss=False)
        img_gen = reshape_patch_back(img_gen, self.args.patch_size)
        pred_y = img_gen[:, -self.args.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()

        return pred_y
