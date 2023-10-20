import torch.nn as nn

from openstl.models import ConvLSTM_Model
from .base_plmethod import Base_plmethod


class ConvLSTM(Base_plmethod):
    r"""ConvLSTM

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    Notice: ConvLSTM requires `find_unused_parameters=True` for DDP training.
    """

    def __init__(self, **args):
        super().__init__(**args)
        self.model = self._build_model(self.hparams)
        self.criterion = nn.MSELoss()
        self.configure_loss(self.criterion)
        

    def _build_model(self, args):
        num_hidden = [int(x) for x in args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return ConvLSTM_Model(num_layers, num_hidden, args)
