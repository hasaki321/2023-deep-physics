import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import dataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import joblib
from torch import optim
import torch.nn as nn

def train_MLP(X_train,y_train,model,lr,alpha,fold,para_path,flag):

    if flag == 0 or flag == 3:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1 - x / 100, last_epoch=-1)
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        batch_size = X_train.shape[0]
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        losses = []
        bar = range(500)
        best = 100
        for idx, epoch in enumerate(bar):
            for i, (inputs, labels) in enumerate(train_loader):
                labels = labels
                optimizer.zero_grad()
                try:
                    out, y = model(inputs)
                    label_loss = criterion(y, labels.unsqueeze(1))
                    recon_loss = criterion(out, inputs)
                    loss = alpha * recon_loss + (1-alpha) * label_loss
                except:
                    y = model(inputs)
                    label_loss = criterion(y, labels.unsqueeze(1))
                    loss = label_loss
                if idx > 100 * 0.5 and loss.item() < best:
                    best = loss.item()
                    torch.save(model.state_dict(), f'{para_path}/model_{flag + 1}_fold_{fold + 1}.ckpt')

                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            scheduler.step()

    else:
        model.fit(X_train, y_train)
        filename = f'{para_path}/model_{flag + 1}_fold_{fold + 1}.pkl'
        joblib.dump(model, filename)

def test_MLP(X_test,y_test,model,fold,para_path,scaler,results_df,l,flag):

    loss = 0
    if flag == 0 or flag == 3:
        model.load_state_dict(torch.load(f'{para_path}/model_{flag + 1}_fold_{fold + 1}.ckpt'))
        model.eval()
        with torch.no_grad():
            try:
                _, test_outputs = model(X_test)
            except:
                test_outputs = model(X_test)
            y_pred = test_outputs.to('cpu')
            y_pred = np.array(y_pred)
        X_test = scaler.inverse_transform(X_test)
        loss = sum((y_test.unsqueeze(1) - test_outputs) ** 2)

    else:
        filename = f'{para_path}/model_{flag + 1}_fold_{fold + 1}.pkl'
        model = joblib.load(filename)
        y_pred = model.predict(X_test)

        loss = (y_pred - y_test) ** 2
        loss = sum(loss)

    fold_df = pd.DataFrame(X_test, columns=[str(i) for i in range(1, l + 1)])
    fold_df['Test Labels'] = y_test
    fold_df['Test Outputs'] = y_pred
    fold_df['Fold'] = fold + 1

    results_df = pd.concat([results_df, fold_df], ignore_index=True)

    return loss,results_df



