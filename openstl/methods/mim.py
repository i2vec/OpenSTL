import torch.nn as nn

from openstl.models import MIM_Model
from .base_plmethod import Base_plmethod


class MIM(Base_plmethod):
    r"""MIM

    Implementation of `Memory In Memory: A Predictive Neural Network for Learning
    Higher-Order Non-Stationarity from Spatiotemporal Dynamics
    <https://arxiv.org/abs/1811.07490>`_.

    """
    
    def __init__(self, **config):
        super().__init__(**config)
        self.model = self._build_model(self.hparams)
        self.criterion = nn.MSELoss()
        self.configure_loss(self.criterion)
        
    
    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return MIM_Model(num_layers, num_hidden, args).to(self.device)
