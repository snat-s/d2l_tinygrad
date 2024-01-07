from tinygrad import Tensor
from tinygrad import nn
from extra.datasets import fetch_mnist
from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd

class CustomDataset(Dataset):
   def __init__(self, X, y):
       self.X = X
       self.y = y

   def __len__(self):
       return len(self.X)

   def __getitem__(self, idx):
       return self.X[idx], self.y[idx]

class Trainer:
    def __init__(self, net=None, net_name: str = 'net'):
        assert net is not None, "The neural network should not be None"
        
        self.optim = nn.optim.Adam(nn.state.get_parameters(net))
        X_train, y_train, X_test, y_test = fetch_mnist(tensors=True)

        self.net_name: str = net_name
        self.X_train = X_train.numpy()
        self.y_train = y_train.numpy()
        self.X_test = X_test.numpy()
        self.y_test = y_test.numpy()
        
        self.val_split = 0.1 # 20% of data will be used for validation
        self.val_size = int(len(self.X_train) * self.val_split)
        
        self.X_val = self.X_train[-self.val_size:]
        self.y_val = self.y_train[-self.val_size:]
        self.X_train = self.X_train[:-self.val_size]
        self.y_train = self.y_train[:-self.val_size]

        self.train_dataset = CustomDataset(self.X_train, self.y_train)
        self.val_dataset = CustomDataset(self.X_val, self.y_val)
        self.test_dataset = CustomDataset(self.X_test, self.y_test)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        self.net = net
        
    def train(self, resize=True, target_size=(224, 224), epochs = 1, plot: bool = True):
        window_size = 10
        lossi = []
        acci = []
        val_lossi = []
        val_acci = []
        

        with Tensor.train():
            for _ in trange(epochs):
                for step, batch in enumerate(tqdm(self.train_loader)):
                    X_batch, y_batch = batch

                    if resize:
                        # Resize and interpolate because alexnet used 224*224 images 8 times bigger than MNIST
                        X_batch = F.interpolate(X_batch, size=target_size, mode='bilinear', align_corners=False)
                        X_batch = Tensor(X_batch.numpy(), requires_grad=False)
                    else:
                        X_batch = Tensor(X_batch.numpy(), requires_grad=False)
                    
                    labels = Tensor(y_batch.numpy())
                    out = self.net(X_batch)
                    loss = out.sparse_categorical_crossentropy(labels)
                    self.optim.zero_grad()
                    loss.backward()

                    self.optim.step()
                    pred = out.argmax(axis=-1)
                    acc = (pred == labels).mean()
                    lossi.append(loss.numpy())
                    acci.append(acc.numpy())

                    if step % 100 == 0:
                        # Validate every 100 steps
                        val_lossi_a = []
                        val_acci_a = []
                        for batch in self.val_loader:
                            X_batch, y_batch = batch
                            if resize:
                                X_batch = F.interpolate(X_batch, size=target_size, mode='bilinear', align_corners=False)
                                X_batch = Tensor(X_batch.numpy(), requires_grad=False)
                            else:
                                X_batch = Tensor(X_batch.numpy(), requires_grad=False)
                        
                            labels = Tensor(y_batch.numpy())
                            out = self.net(X_batch)
                            loss = out.sparse_categorical_crossentropy(labels)
                            pred = out.argmax(axis=-1)
                            acc = (pred == labels).mean()
                            val_lossi_a.append(loss.numpy())
                            val_acci_a.append(acc.numpy())
                        val_lossi.append(np.mean(val_lossi_a))
                        val_acci.append(np.mean(val_acci_a))

            if plot:
                lossi_series = pd.Series(lossi)
                acci_series = pd.Series(acci)
                val_lossi_series = pd.Series(val_lossi)
                val_acci_series = pd.Series(val_acci)
      
                lossi_avg = lossi_series.rolling(window=window_size).mean()
                acci_avg = acci_series.rolling(window=window_size).mean()
                val_lossi_avg = val_lossi_series.rolling(window=window_size).mean()
                val_acci_avg = val_acci_series.rolling(window=window_size).mean()
                #x_val = np.arange(len(lossi_avg)) # + len(lossi_avg) - window_size
                plt.figure()
                plt.plot(lossi_avg, label='Training Loss')
                plt.plot(acci_avg, label='Training Accuracy')
                plt.title("Training Loss & Accuracy over time")
                plt.legend()
                plt.savefig(f'./assets/training_{self.net_name}.png')
                
                plt.figure()
                plt.plot(val_lossi_avg, label='Validation Loss')
                plt.plot(val_acci_avg, label='Validation Accuracy')
                plt.legend()
                plt.savefig(f'./assets/validation_{self.net_name}.png')
                
            state_dict = nn.state.get_state_dict(self.net)
            nn.state.safe_save(state_dict, f"./models/{self.net_name}.safetensor")
                
            return self.net