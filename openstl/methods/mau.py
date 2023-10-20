import time
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm

from openstl.models import MAU_Model
from openstl.utils import reduce_tensor, schedule_sampling
from .base_plmethod import Base_plmethod




class MAU(Base_plmethod):
    r"""MAU

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, **config):
        super().__init__(**config)
        self.model = self._build_model(self.hparams)
        self.criterion = nn.MSELoss()
        self.configure_loss(self.criterion)

    def _build_model(self, args):
        num_hidden = [int(x) for x in args['num_hidden'].split(',')]
        num_layers = len(num_hidden)
        return MAU_Model(num_layers, num_hidden, args).to(self.device)

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model."""
        _, img_channel, img_height, img_width = self.args.in_shape

        # preprocess
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            self.args.total_length - self.args.pre_seq_length - 1,
            img_height // self.args.patch_size,
            img_width // self.args.patch_size,
            self.args.patch_size ** 2 * img_channel)).to(self.device)

        img_gen, _ = self.model(test_ims, real_input_flag, return_loss=False)
        pred_y = img_gen[:, -self.args.aft_seq_length:, :]

        return pred_y