import time
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.core.optim_scheduler import get_optim_scheduler
from openstl.models import CrevNet_Model
from openstl.utils import reduce_tensor
from .base_plmethod import Base_plmethod


class CrevNet(Base_plmethod):
    r"""CrevNet

    Implementation of `Efficient and Information-Preserving Future Frame Prediction
    and Beyond <https://openreview.net/forum?id=B1eY_pVYvB>`_.
    """

    def __init__(self, **args):
        super().__init__(**args)
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.configure_loss(self.criterion)
        

    def _build_model(self, **args):
        return CrevNet_Model(**args).to(self.device)

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        input = torch.cat([batch_x, batch_y], dim=1)
        pred_y, _ = self.model(input, training=False, return_loss=False)
        return pred_y