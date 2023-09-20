import torch 
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np

class dataset(Dataset):
    def __init__(self,data,Type,N_size,input_size,scaler):
        super().__init__()
        index = np.lexsort((data[:,2],))
        data = data[index]
        data = np.array(data)
        if N_size == '<126':
            data = data[np.where(data[:,2]<126)]
        else:
            data = data[np.where(data[:,2]>=126)]
        
        if Type == 'odd':
            data = data[np.intersect1d(np.where(data[:,2]%2==1),np.where(data[:,1]%2==1) )]
        elif Type == 'even':
            data = data[np.intersect1d(np.where(data[:,2]%2==0),np.where(data[:,1]%2==0))]
        else:
            data = data[np.where((data[:,2]+data[:,1])%2==1)]
        
        
        self.data = data[:,:input_size].astype(np.float32)
        if scaler:
            self.data = scaler.fit_transform(self.data)
        
        self.target = data[:,-1].astype(np.float32)
        
        
    def __getitem__(self,idx):
        return self.data[idx],self.target[idx]
    
    def __len__(self):
        return len(self.data)
    
    
def get_Kfold(file,k,shuffle=True):
    data = pd.read_excel(file,engine="openpyxl")
    data = np.array(data)
    if shuffle:
        np.random.shuffle(data) 
    num = len(data)//k
    return [(np.concatenate((data[0:num*i],data[num*(i+1):len(data)])),data[num*i:num*(i+1)]) for i in range(k)]