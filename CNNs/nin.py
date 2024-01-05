# Network in Network solve the problem of having to much linearity by using a 1x1 conv as a replacement
# for a fully connected network, we can add non-linearity and it is more efficient.

import matplotlib.pyplot as plt
from tinygrad import nn
from tinygrad.helpers import flatten
from tinygrad import Tensor
from tinygrad.jit import TinyJit
from tinygrad.helpers import Timing
from extra.datasets import fetch_mnist
from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F

class NiNBlock:
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        self.layers = [
                    nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding,), Tensor.relu,
                    nn.Conv2d(out_channels, out_channels, kernel_size=1), Tensor.relu,
                    nn.Conv2d(out_channels, out_channels, kernel_size=1), Tensor.relu,
                ]

    def __call__(self, x):
        return x.sequential(self.layers)

class NiN:
   def __init__(self, num_classes=10):
       input_size, output_size = 1, 1
       stride = (input_size//output_size)
       kernel_size = input_size - (output_size-1)*stride
       self.layers = [
               NiNBlock(1,96,kernel_size=11, strides=4,padding=0),
               lambda x: x.max_pool2d(kernel_size=3, stride=2),
               NiNBlock(96,256,kernel_size=5, strides=1,padding=2),
               lambda x: x.max_pool2d(kernel_size=3, stride=2),
               NiNBlock(256,384,kernel_size=3, strides=1,padding=1),
               lambda x: x.max_pool2d(kernel_size=3, stride=2),
               Tensor.dropout,
               NiNBlock(384, num_classes,kernel_size=3, strides=1,padding=1),
               lambda x: x.avg_pool2d(kernel_size=5), # There is no necessity for an adaptive avg_pool, just use a 5x5 kernel
               lambda x: x.flatten(1),
               ]

   def __call__(self, x):
       return x.sequential(self.layers)

net = NiN()
state_dict = nn.state.safe_load("NiN.safetensor")
nn.state.load_state_dict(net, state_dict)

optim = nn.optim.Adam(nn.state.get_parameters(net))
X_train, y_train, X_test, y_test = fetch_mnist(tensors=True)

X_train = X_train.numpy()
y_train = y_train.numpy()
X_test = X_test.numpy()
y_test = y_test.numpy()

if False:
    with Tensor.train():
        lossi = []
        for step in trange(1000):
            samp = np.random.randint(0, X_train.shape[0], size=(32))
            batch = X_train[samp]
            # Define the target size (224x224)
            target_size = (224, 224)
            # Resize and interpolate because alexnet used 224*224 images 8 times bigger than MNIST
            batch = F.interpolate(torch.tensor(batch), size=target_size, mode='bilinear', align_corners=False)
            batch = Tensor(batch.numpy(), requires_grad=False)
            labels = Tensor(y_train[samp])
            out = net(batch)
            loss = out.sparse_categorical_crossentropy(labels)
            optim.zero_grad()
            loss.backward()

            optim.step()
            pred = out.argmax(axis=-1)
            acc = (pred == labels).mean()
            lossi.append(loss.numpy())
            if step % 100 == 0:
                print(f"step {step}, {loss.numpy()=}, {acc.numpy()=}")
        plt.plot(lossi)
        state_dict = nn.state.get_state_dict(net)
        nn.state.safe_save(state_dict, "NiN.safetensor")
        
parameters = nn.state.get_parameters(net)
n_parameters = 0
for layer in parameters:
    n_parameters += len(layer.numpy())
print(n_parameters)

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
