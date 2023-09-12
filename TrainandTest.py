import torch
import numpy as np
import torch.nn as nn

def train_MLP(train_loader,model,criterion,optimizer,epoch,fold,lr_scheduler):
    losses = []
    bar = range(epoch)
    best = 100
    for idx,epoch in enumerate(bar):
        for i, (inputs, labels) in enumerate(train_loader):
            labels = labels 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            if loss.item()<best and idx>epoch*0.95:
                best = loss.item()
                torch.save(model.state_dict(), f'./model_fold_{fold + 1}.ckpt')
                print(f"loss:{loss.item()} saving...")

        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    lr_scheduler.step()

def test_MLP(test_loader,model,fold,criterion):
  model.load_state_dict(torch.load(f'./model_fold_{fold + 1}.ckpt'))
  model.eval()
  test_losses = []
  with torch.no_grad():  
    for i, (inputs, labels) in enumerate(test_loader):
        labels = labels 
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
      
    test_losses.append(loss.item())
    test_loss = sum(test_losses)/len(test_loss)
    rmse = torch.sqrt(test_loss)
    print(f'Test Loss: {test_loss:.10f}, Test RMSE: {rmse:.10f}') 
   