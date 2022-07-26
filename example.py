from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch
import torch.nn as nn
import pdb


class _CN(_BatchNorm):
    def __init__(self, target, eps = 1e-5, momentum = 0.1, affine=True):
        num_features = target.num_features
        super(_CN, self).__init__(num_features, eps, momentum, affine=True)
        self.running_mean = target.running_mean
        self.running_var = target.running_var
        
        self.weight = target.weight
        self.bias = target.bias

        self.N = num_features
        self.setG()

    def setG(self):
        pass

    def forward(self, input):
        out_gn = F.group_norm(input, self.G, None, None, self.eps)
        out = F.batch_norm(out_gn, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)
        return out

class CN4(_CN):
    def setG(self):
        self.G = 4

class CN8(_CN):
    def setG(self):
        self.G = 8

class CN16(_CN):
    def setG(self):
        self.G = 16

class CN32(_CN):
    def setG(self):
        self.G = 32

class CN64(_CN):
    def setG(self):
        self.G = 64
		
###########################################################################
## Utility function to replace BN on existing models
###########################################################################
def replace_bn(module, name, nl):
	for attr_str in dir(module):
		target_attr = getattr(module, attr_str)
		if type(target_attr) == torch.nn.BatchNorm2d:
			new_bn = nl(target_attr)
			setattr(module, attr_str, new_bn)
	for name, icm in module.named_children():
		if type(icm) == torch.nn.BatchNorm2d:
			new_bn = nl(icm)
			setattr(module, name, new_bn)
		replace_bn(icm, name, nl_fn)

###########################################################################
## Example
###########################################################################
from torchvision.models import resnet18

net = resnet18(pretrained=False)
new_nl = CN8
replace_bn(net, 'model', new_nl)