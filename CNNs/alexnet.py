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

class AlexNet:
    def __init__(self, num_classes=10):
       self.layers = [
                nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4), Tensor.relu,
                lambda x : x.max_pool2d(kernel_size=3, stride=2),
                lambda x: x.max_pool2d(kernel_size=(2,2), stride=2),
                nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), Tensor.relu,
                lambda x : x.max_pool2d(kernel_size=3, stride=2),
                #lambda x : print(x.shape)
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), Tensor.relu,
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), Tensor.relu,
                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), Tensor.relu,
                lambda x: x.max_pool2d(kernel_size=3, stride=2),
                #lambda x: print(x.shape),
                lambda x: x.flatten(1),
                nn.Linear(256*2*2, 4096), Tensor.relu,
                nn.Linear(4096, 4096), Tensor.relu,
                nn.Linear(4096, num_classes)
               ] 
    #@TinyJit
    def __call__(self, x): 
        return x.sequential(self.layers)#.realize()

net = AlexNet()
optim = nn.optim.Adam(nn.state.get_parameters(net))
X_train, y_train, X_test, y_test = fetch_mnist(tensors=True)

X_train = X_train.numpy()
y_train = y_train.numpy()
X_test = X_test.numpy()
y_test = y_test.numpy()

state_dict = nn.state.safe_load("AlexNet.safetensor")
nn.state.load_state_dict(net, state_dict)

if False:
    with Tensor.train():
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

            if step % 100 == 0:
                print(f"step {step}, {loss.numpy()=}, {acc.numpy()=}")

        state_dict = nn.state.get_state_dict(net)
        nn.state.safe_save(state_dict, "AlexNet.safetensor")

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
