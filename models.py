import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size,Linear_size):
        super(MLP, self).__init__()
        self.input_size = input_size

        self.fnn1 = nn.Sequential(
            nn.Linear(input_size,Linear_size),
            nn.ELU(),
            nn.BatchNorm1d(Linear_size),
            nn.Linear(Linear_size,1),
        )

    def forward(self, x):
        out = self.fnn1(x)
        return out

def KlLoss(mu,logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class MLP(nn.Module):
    def __init__(self, input_size,hid_feature,Linear_size1,Linear_size2):
        super().__init__()
        self.input_size = input_size
        self.hid_feature = hid_feature

        self.encoder = nn.Sequential(
            nn.Linear(input_size,Linear_size1),
            nn.ELU(),
            nn.BatchNorm1d(Linear_size1),
            nn.Linear(Linear_size1,Linear_size2),
            nn.ELU(),
            nn.Linear(Linear_size2,hid_feature),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hid_feature,Linear_size2),
            nn.ReLU(),
            nn.BatchNorm1d(Linear_size2),
            nn.Linear(Linear_size2,Linear_size1),
            nn.ELU(),
            nn.Linear(Linear_size1,input_size),
        )
        
        self.predicter = nn.Sequential(
            nn.Linear(hid_feature,Linear_size2),
            nn.ReLU(),
            nn.BatchNorm1d(Linear_size2),
            nn.Linear(Linear_size2,Linear_size1),
            nn.ELU(),
            nn.BatchNorm1d(Linear_size1),
            nn.Linear(Linear_size1,1),
        )
    
    def reparameterize(self,hid):
        mu,logvar = hid[:,hid_feature//2:],hid[:,:hid_feature//2]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return mu,logvar,z

    def forward(self, x):
        hid = self.encoder(x)
        out = self.decoder(hid)
        y = self.predicter(hid)
        return out,y
    
