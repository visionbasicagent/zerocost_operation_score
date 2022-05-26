from random import choice
import torch.nn as nn
import torch
from collections import OrderedDict

candidate_OP = ['id', 'ir_3x3_t3', 'ir_5x5_t6']
OPS = OrderedDict()
OPS['id'] = lambda inp, oup, stride: Identity(inp=inp, oup=oup, stride=stride)
OPS['ir_3x3_t3'] = lambda inp, oup, stride: InvertedResidual(inp=inp, oup=oup, t=3, stride=stride, k=3)
OPS['ir_5x5_t6'] = lambda inp, oup, stride: InvertedResidual(inp=inp, oup=oup, t=6, stride=stride, k=5)


class Identity(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Identity, self).__init__()
        if stride != 1 or inp != oup:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.downsample = None

    def forward(self, x, block_input=False):
        if block_input:
            x = x*0
        if self.downsample is not None:
            x = self.downsample(x)
        return x



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, use_se=False, **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.t = t
        self.k = k
        self.use_se = use_se
        assert stride in [1, 2]
        hidden_dim = round(inp * t)
        if t == 1:
            self.conv = nn.Sequential(
                # dw            
                nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, 
                              bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, 
                              bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.use_shortcut = inp == oup and stride == 1

    def forward(self, x, block_input=False):
        if block_input:
            x = x*0
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, choice_block_id):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in candidate_OP:
            op = OPS[primitive](C_in, C_out, stride)
            self._ops.append(op)
        
        self.choice_block_id = choice_block_id 

    def forward(self, x, weights):
        #print(weights)
        weights = weights[self.choice_block_id]
        #print(weights, self.choice_block_id)
        ret = sum((w * op(x, block_input=True) if w == 0 else w * op(x) for w, op in zip(weights, self._ops)))
        return ret

class Network(nn.Module):
    def __init__(self, num_classes=10, stages=[2, 3, 3], init_channels=32):
        super(Network, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        self.features = nn.ModuleList()
        self.stages = stages
        channels = init_channels
        choice_block_id = 0
        for stage in stages:
            for idx in range(stage):
                if idx == 0:
                    # stride = 2 
                    self.features.append(MixedOp(channels, channels*2, 2, choice_block_id))
                    channels *= 2
                else:
                    self.features.append(MixedOp(channels, channels, 1, choice_block_id))
                choice_block_id+=1
        self.out = nn.Sequential(
            nn.Conv2d(channels, 1280, kernel_size=1, bias=False, stride=1),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(1280, num_classes)


    def forward(self, x, weights):
        x = self.stem(x)
        for idx in range(sum(self.stages)):
            x=self.features[idx](x, weights)

        x = self.out(x)
        out = self.classifier(x.view(x.size(0), -1))
        return out