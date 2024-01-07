# The first result of the convolutions
from tinygrad import nn
from tinygrad.helpers import flatten
from tinygrad import Tensor
from tinygrad.jit import TinyJit
from tinygrad.helpers import Timing
from extra.datasets import fetch_mnist
from tqdm import trange
import numpy as np
from trainer import Trainer

class LeNet:
    def __init__(self, num_classes=10):
        self.layers = [
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), Tensor.sigmoid,
                lambda x: x.max_pool2d(kernel_size=(2,2), stride=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), Tensor.sigmoid,
                lambda x: x.max_pool2d(kernel_size=(2,2), stride=2),
                #lambda x: print(x.shape),
                lambda x: x.flatten(1),
                nn.Linear(16*5*5, 120), Tensor.sigmoid,
                nn.Linear(120, 84), Tensor.sigmoid, 
                nn.Linear(84, num_classes)
                ]
    #@TinyJit
    def __call__(self, x):
        return x.sequential(self.layers)#.realize()

net = LeNet()
X_train, y_train, X_test, y_test = fetch_mnist(tensors=True)
X_test = X_test.numpy()
y_test = y_test.numpy()

train = False

if train:
    trainer = Trainer(net=net, net_name='LeNetMax')
    net = trainer.train(resize=False)
else:
    state_dict = nn.state.safe_load("./models/LeNetMax.safetensor")
    nn.state.load_state_dict(net, state_dict)

@TinyJit
def jit(x):
    return net(x).realize()

with Timing("Time: "):
    avg_acc = 0
    for step in trange(1000):
        samp = np.random.randint(0, X_test.shape[0], size=(64))
        batch = Tensor(X_test[samp], requires_grad=False)
        labels = y_test[samp]
        out = jit(batch)
        preds = out.argmax(axis=1).numpy()
        avg_acc += (labels == preds).mean()
    print(f'Avg accuracy {avg_acc/1000}') 
