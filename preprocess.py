import torch 
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class dataset(Dataset):
    def __init__(self, data, Type, N_size, input_size):
        super().__init__()
        index = np.lexsort((data[:, 2],))
        data = data[index]
        data = np.array(data)
        if N_size == '<=126':
            data = data[np.where(data[:, 2] <= 126)]
        else:
            data = data[np.where(data[:, 2] > 126)]

        if Type == 'odd':
            data = data[np.intersect1d(np.where(data[:, 2] % 2 == 1), np.where(data[:, 1] % 2 == 1))]
        elif Type == 'even':
            data = data[np.intersect1d(np.where(data[:, 2] % 2 == 0), np.where(data[:, 1] % 2 == 0))]
            data = np.delete(data, [4, 5, 6], axis=1)
        else:
            data = data[np.where((data[:, 2] + data[:, 1]) % 2 == 1)]

        self.data = data[:, :input_size].astype(np.float32)

        self.target = data[:, -1]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def __len__(self):
        return len(self.data)

def data_deal(X_train,X_test,y_train,y_test,scaler,input_size,flag):
    X_train = np.concatenate(X_train).reshape(len(X_train),input_size)
    X_test = np.concatenate(X_test).reshape(len(X_test), input_size)

    if flag == 0 or flag == 3:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train,X_test,y_train,y_test,scaler


