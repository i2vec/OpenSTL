import torch.nn as nn

from openstl.models import PredRNNpp_Model
from .base_plmethod import Base_plmethod


class PredRNNpp(Base_plmethod):
    r"""PredRNN++

    Implementation of `PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma
    in Spatiotemporal Predictive Learning <https://arxiv.org/abs/1804.06300>`_.

    """

    def __init__(self, **config):
        super().__init__(**config)
        self.model = self._build_model(self.hparams)
        self.criterion = nn.MSELoss()
        self.configure_loss(self.criterion)


    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PredRNNpp_Model(num_layers, num_hidden, args).to(self.device)
