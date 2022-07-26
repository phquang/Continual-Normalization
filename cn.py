from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch
import torch.nn as nn



class CN(_BatchNorm):
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1, G=32, affine=True):
        super(_CN, self).__init__(num_features, eps, momentum, affine=True)
        self.G = G

    def forward(self, input):
        out_gn = F.group_norm(input, self.G, None, None, self.eps)
        out = F.batch_norm(out_gn, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)
        return out

