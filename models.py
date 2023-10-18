import torch.nn as nn
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def get_models(model,input_size,epsilon):
    if model=='SVR':
        SVR_model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.01, epsilon=epsilon))
        return SVR_model
    elif model == 'AE_MLP':
        return AE_MLP(input_size)
    
    elif model == 'MLP':
        return MLP(input_size)

class AE_MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size,8),
            nn.ELU(),
            nn.BatchNorm1d(8),
            nn.Linear(8,16),
            nn.ELU(),
            nn.Linear(16,32),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16,8),
            nn.ELU(),
            nn.Linear(8,input_size),
        )
        
        self.predicter = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16,8),
            nn.ELU(),
            nn.BatchNorm1d(8),
            nn.Linear(8,1),
        )
    
    def forward(self, x):
        hid = self.encoder(x)
        out = self.decoder(hid)
        y = self.predicter(hid)
        return out,y
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size,8),
            nn.ELU(),
            nn.BatchNorm1d(8),
            nn.Linear(8,16),
            nn.ELU(),
            nn.Linear(16,32),
        )
        
        self.predicter = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16,8),
            nn.ELU(),
            nn.BatchNorm1d(8),
            nn.Linear(8,1),
        )
    
    def forward(self, x):
        hid = self.encoder(x)
        y = self.predicter(hid)
        return y
        

