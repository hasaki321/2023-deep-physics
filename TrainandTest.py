import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import dataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler
def loss_fn(criterion,alpha,inputs,outputs,labels):
    
    #let the length of output determin which loss to return 
    if len(outputs) == 2:
        mse = criterion(outputs[0], inputs)
        loss = alpha*criterion(outputs[1], labels.unsqueeze(1)) + (1-alpha)*mse
    else:
        loss = criterion(outputs, labels.unsqueeze(1))
        mse = loss
    #alse return mse
    return loss,mse
    
def pre_training(model,criterion,alpha,optimizer,epoch,scheduler,para_path,device,flag):
    if flag == 5 :
        scaler = MinMaxScaler()
        pre_traindata = pd.read_excel('./data/test 3.xlsx',engine="openpyxl")
        pre_traindata = np.array(pre_traindata)
        pre_traindata = dataset(pre_traindata,'even','>126',5, scaler)

        best = 100
        for idx in range(epoch):
            for i, (inputs, labels) in enumerate(pre_traindata):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # change into a loss function
                loss, _ = loss_fn(criterion, alpha, inputs, outputs, labels)

                if loss.item() < best and idx > epoch * 0.5:
                    best = loss.item()
                    torch.save(model.state_dict(), f'{para_path}/model.ckpt')
 #                   print(f"loss:{loss.item()} saving...")

            loss.backward()
            optimizer.step()

        scheduler.step()

    model.load_state_dict(torch.load(f'{para_path}/model.ckpt'))

def train_MLP(train_loader,model,criterion,alpha,optimizer,epoch,fold,scheduler,para_path,device,flag):
    losses = []
    best = 100
    for idx in range(epoch):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            #change into a loss function
            loss,_ = loss_fn(criterion,alpha,inputs,outputs,labels)
            
            if loss.item()<best and idx>epoch*0.5:
                best = loss.item()
                torch.save(model.state_dict(), f'{para_path}/model_{flag + 1}_fold_{fold + 1}.ckpt')
#                print(f"loss:{loss.item()} saving...")

        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    if flag != 1:
        scheduler.step()
    else:
        scheduler.step(loss)

def test_MLP(test_loader,model,fold,criterion,alpha,plot,para_path,device,flag):
    model.load_state_dict(torch.load(f'{para_path}/model_{flag + 1}_fold_{fold + 1}.ckpt'))
    model.eval()
    test_mses,test_losses,test_output,test = [],[],[],[]
    
    with torch.no_grad():  
        for i,(inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            test.append(labels.unsqueeze(1))
            test_output.append(outputs[1] if len(outputs)==2 else outputs)
            loss,mse = loss_fn(criterion,alpha,inputs,outputs,labels)
            
            #separate loss and mse here
            test_mses.append(mse.item())
            test_losses.append(loss.item())

        total_len_mse = sum(test_mses)
        mse = total_len_mse/len(test_mses)
        rmse = sum(test_mses)/len(test_mses)**0.5
        print(f'Test Loss: {mse:.5f}, Test RMSE: {rmse:.5f}')
    if plot:
        test = np.array(torch.stack(test).cpu()).reshape(-1)
        test_output = np.array(torch.stack(test_output).cpu()).reshape(-1)
        plt.scatter(test,test)
        plt.scatter(test,test_output)
        plt.show()
    return total_len_mse,rmse
   
