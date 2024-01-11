# This is like a bomb to a simple problem

from typing import Any
from tinygrad import nn
from tinygrad import Tensor
from tinygrad.jit import TinyJit
from tinygrad.helpers import Timing
from extra.datasets import fetch_mnist
import numpy as np
import torch
import torch.nn.functional as F
from trainer import Trainer
from tqdm import tqdm, trange

class Residual:
    def __init__(self, in_channels, out_channels=None, use_1x1conv = False, strides = 1):
        if out_channels is None: out_channels = in_channels
        #print(in_channels, out_channels)
        self.c1 = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides),
            nn.BatchNorm2d(out_channels),
            Tensor.relu,
        ]

        self.c2 = [ 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        ]
        if use_1x1conv:
            self.c3 = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)]
        else:
            self.c3 = None
    
    def __call__(self, x: Tensor) -> Tensor:
        y = x.sequential(self.c1).sequential(self.c2)
        
        if self.c3:
            x = x.sequential(self.c3)
        y = y + x
        return y.relu()
    

# I actually don't love this but it is the way of the book
# Classes used only when necessary
class ResNet:
    @staticmethod
    def block(num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(int(num_channels/2),num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return blk

    def __init__(self, arch, num_classes=10):
        # Block 1
        self.layers = [
            nn.Conv2d(1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), Tensor.relu,
            lambda x : x.max_pool2d(kernel_size=3, stride=2),
        ]

        # Arch
        for i, b in enumerate(arch):
            blk = self.block(num_residuals=b[0], num_channels=b[1], first_block=(i==0))
            self.layers += blk# + [lambda x : print(x.shape)]

        # Last block
        last_block = [lambda x : x.avg_pool2d(kernel_size=3), lambda x: x.flatten(1),
                nn.Linear(512, num_classes)]
        self.layers += last_block

class ResNet18(ResNet):
    def __init__(self):
       super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)))

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)

net = ResNet18()
X = Tensor.randn((1, 1, 96, 96))

train = False 
target_size = (96,96)
print(net(X).shape)

if train:
   trainer = Trainer(net=net,net_name='resnet18')
   net = trainer.train(resize=True, target_size=target_size)
else:
    state_dict = nn.state.safe_load("./models/resnet18.safetensor")
    nn.state.load_state_dict(net, state_dict)


_, _, X_test, y_test = fetch_mnist(tensors=True)

X_test = X_test.numpy()
y_test = y_test.numpy()


@TinyJit
def jit(x):
    return net(x).realize()

with Timing("Time: "):
    avg_acc = 0
    for step in trange(1000):
        samp = np.random.randint(0, X_test.shape[0], size=(64))
        batch = X_test[samp]

        # Resize and interpolate the tensor using F.interpolate
        batch = F.interpolate(torch.tensor(batch), size=target_size, mode='bilinear', align_corners=False)
        batch = Tensor(batch.numpy(), requires_grad=False)
        labels = y_test[samp]
        out = jit(batch)
        preds = out.argmax(axis=1).numpy()
        avg_acc += (labels == preds).mean()
    print(f'Avg accuracy {avg_acc/1000}') 
