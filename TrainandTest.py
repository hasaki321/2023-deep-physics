import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def train_MLP(train_loader,model,criterion,optimizer,epoch,fold,lr_scheduler,para_path):
    losses = []
    bar = range(epoch)
    best = 100
    for idx,epoch in enumerate(bar):
        for i, (inputs, labels) in enumerate(train_loader):
            labels = labels 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            if loss.item()<best and idx>epoch*0.98:
                best = loss.item()
                torch.save(model.state_dict(), f'{para_path}/model_fold_{fold + 1}.ckpt')
                print(f"loss:{loss.item()} saving...")

        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    lr_scheduler.step()

def test_MLP(test_loader,model,fold,criterion,plot,para_path):
    model.load_state_dict(torch.load(f'{para_path}/model_fold_{fold + 1}.ckpt'))
    model.eval()
    test_losses = []
    test_output = []
    test = []
    with torch.no_grad():  
        for i,(inputs, labels) in enumerate(test_loader):
            labels = labels 
            outputs = model(inputs)
            test.append(labels.unsqueeze(1))
            test_output.append(outputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            test_losses.append(loss.item())
        total_mse = sum(test_losses)
        test_loss = sum(test_losses)/len(test_losses)
        rmse = test_loss**0.5
        print(f'Test Loss: {test_loss:.10f}, Test RMSE: {rmse:.10f}') 
    if plot:
        test = np.array(torch.stack(test)).squeeze(0).squeeze(1)
        test_output = np.array(torch.stack(test_output)).squeeze(0).squeeze(1)
        plt.scatter(test,test)
        plt.scatter(test,test_output)
        plt.show()
    return total_mse
   
