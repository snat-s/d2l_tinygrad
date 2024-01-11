# VGG is the first family of CNNs
# The part of VGG is that it still needs a lot for the fully connected network at the end
from tinygrad import nn
from tinygrad import Tensor
from tinygrad.jit import TinyJit
from tinygrad.helpers import Timing
from extra.datasets import fetch_mnist
from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F
from trainer import Trainer

class VGGBlock:
    def __init__(self, in_channels, out_channels, num_convs):
        self.layers = []
        for _ in range(num_convs):
            self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
            self.layers.append(Tensor.relu)
            in_channels=out_channels
        self.layers.append(lambda x : x.max_pool2d(kernel_size=2, stride=2))

    def __call__(self, x):
        return x.sequential(self.layers)

class VGG:
    def __init__(self, architecture, num_classes=10):
        conv_blocks = []
        for (num_convs, in_channels, out_channels) in architecture:
            conv_blocks.append(VGGBlock(in_channels, out_channels, num_convs))
        self.layers = [*conv_blocks,  lambda x: x.flatten(1),
                       nn.Linear(128*7*7, 4096), Tensor.relu,
                       nn.Linear(4096, 4096), Tensor.relu,
                       nn.Linear(4096, num_classes)
                       ]

    def __call__(self, x):
        return x.sequential(self.layers)
# Original
#net = VGG(architecture=((1,1,64), (1,64,128), (2, 128, 256), (2, 256, 512), (2, 512, 512)))
# Implemented and trained in d2l
net = VGG(architecture=((1,1,16), (1,16,32),(2,32,64), (2,64,128),(2,128,128)))
X_train, y_train, X_test, y_test = fetch_mnist(tensors=True)

X_test = X_test.numpy()
y_test = y_test.numpy()

train = False

if train:
    trainer = Trainer(net=net, net_name='VGG11')
    net = trainer.train(resize=True, target_size=(224, 224))
else:
    state_dict = nn.state.safe_load("./models/VGG11.safetensor")
    nn.state.load_state_dict(net, state_dict)


@TinyJit
def jit(x):
    return net(x).realize()

with Timing("Time: "):
    avg_acc = 0
    for step in trange(1000):
        samp = np.random.randint(0, X_test.shape[0], size=(64))
        batch = X_test[samp]
        target_size = (224, 224)

        # Resize and interpolate the tensor using F.interpolate
        batch = F.interpolate(torch.tensor(batch), size=target_size, mode='bilinear', align_corners=False)
        batch = Tensor(batch.numpy(), requires_grad=False)
        labels = y_test[samp]
        out = jit(batch)
        preds = out.argmax(axis=1).numpy()
        avg_acc += (labels == preds).mean()
    print(f'Avg accuracy {avg_acc/1000}') 
