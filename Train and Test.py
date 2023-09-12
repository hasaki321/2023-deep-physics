def train_MLP(train_loader,model,criterion,optimizer,device,epoch,fold):
  losses = []
  bar = range(num_epochs)
  best = 100
  for idx,epoch in enumerate(bar):
    for i, (inputs, labels) in enumerate(train_loader):
        labels = labels 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        if loss.item()<best and idx>num_epochs*save_rate:
            best = loss.item()
            torch.save(model.state_dict(), f'./model_fold_{fold + 1}.ckpt')
            print(f"loss:{loss.item()} saving...")
  
        loss.backward()
        optimizer.step()
  
    losses.append(loss.item())
    lr_scheduler.step()

def train_AEMLP(train_loader,model,criterion,optimizer,device,epoch,fold):
  losses = []
  bar = range(num_epochs)
  best = 100
  for idx,epoch in enumerate(bar):
    for i, (inputs, labels) in enumerate(train_loader):
        labels = labels 
        optimizer.zero_grad()
        out,y = model(inputs)
        label_loss = criterion(y, labels.unsqueeze(1))
        recon_loss = criterion(out, inputs)
        loss = alpha*recon_loss + beta*label_loss
        if  idx>num_epochs*save_rate and loss.item()<best:
            best = loss.item()
            torch.save(model.state_dict(), f'./model_fold_{fold + 1}.ckpt')
            print(f"loss:{loss.item()} saving...")
        loss.backward()
        optimizer.step()
  
    losses.append(loss.item())
    lr_scheduler.step()

def protrain_AEMLP(train_loader,model,criterion,optimizer,device,epoch):
  losses = []
  best = 100
  bar = range(num_epochs)
  for idx,epoch in enumerate(bar):
      for i, (inputs, labels) in enumerate(train_loader):
          labels = labels 
          optimizer.zero_grad()
          out,y = model(inputs)
          label_loss = criterion(y, labels.unsqueeze(1))
          recon_loss = criterion(out, inputs)
          loss = alpha*recon_loss + beta*label_loss
          if  idx>num_epochs*save_rate and loss.item()<best:
              best = loss.item()
              torch.save(model.state_dict(), f'./model.ckpt')
              print(f"loss:{loss.item()} saving...")
          loss.backward()
          optimizer.step()
  
      losses.append(loss.item())
      lr_scheduler.step()

def test_AEMLP(test_loader,model,device,fold):
  model.load_state_dict(torch.load(f'./model_fold_{fold + 1}.ckpt'))
  model.eval()
  
  with torch.no_grad():
      _,test_outputs = model(test_loader) #呃呃呃
      y_pred = test_outputs.to('cpu')
      test_loss = criterion(test_outputs, y_test_tensor.unsqueeze(1))
      rmse = torch.sqrt(test_loss)  # 计算均方根误差
      y_out = (y_pred - y_test_tensor.numpy().reshape(len(y_test_tensor),1))/y_test_tensor.numpy().reshape(len(y_test_tensor),1)
      y_out = y_out.abs() < 0.1
      accuracy = y_out.sum()/len(y_out)
      print(f'Test Loss: {test_loss:.10f}, Test RMSE: {rmse:.10f},Accuracy: {accuracy:.10f}')

def test_MLP(test_loader,model,device,fold):
  model.load_state_dict(torch.load(f'./model_fold_{fold + 1}.ckpt'))
  model.eval()
  
  with torch.no_grad():
      test_outputs = model(test_loader)
      y_pred = test_outputs.to('cpu')
      test_loss = criterion(test_outputs, y_test_tensor.unsqueeze(1))
      rmse = torch.sqrt(test_loss)  # 计算均方根误差
      y_out = (y_pred - y_test_tensor.numpy().reshape(len(y_test_tensor),1))/y_test_tensor.numpy().reshape(len(y_test_tensor),1)
      y_out = y_out.abs() < 0.1
      accuracy = y_out.sum()/len(y_out)
      print(f'Test Loss: {test_loss:.10f}, Test RMSE: {rmse:.10f},Accuracy: {accuracy:.10f}')
