import torch.nn as nn

from openstl.models import E3DLSTM_Model
from .base_plmethod import Base_plmethod


class E3DLSTM(Base_plmethod):
    r"""E3D-LSTM

    Implementation of `EEidetic 3D LSTM: A Model for Video Prediction and Beyond
    <https://openreview.net/forum?id=B1lKS2AqtX>`_.

    """
    def __init__(self, **args):
        super().__init__(**args)
        self.model = self._build_model(self.hparams)
        self.criterion = nn.MSELoss()
        self.configure_loss(self.criterion)
        

    def _build_model(self, args):
        num_hidden = [int(x) for x in args['num_hidden'].split(',')]
        num_layers = len(num_hidden)
        return E3DLSTM_Model(num_layers, num_hidden, args)
