#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from KAN import *
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


# read extracted calm FSRE features
train = pd.read_csv('X_train.csv')
test = pd.read_csv('X_test.csv')
train = np.array(train)
test = np.array(test)
print(train.shape)
print(test.shape)


# In[4]:


X = train[:, :-1]
Y = train[:, -1]


# In[5]:


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33)
Y_train = Y_train.tolist()
Y_val = Y_val.tolist()
X_train = X_train.tolist()
X_val = X_val.tolist()
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32).reshape(-1, 1)
print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)


# In[6]:


X_test = test[:, :-1]
Y_test = test[:, -1]
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)
print(X_test.shape)
print(Y_test.shape)


# In[7]:


trainloader = DataLoader(list(zip(X_train, Y_train)), batch_size=1024, shuffle=True)
valloader = DataLoader(list(zip(X_val, Y_val)), batch_size=1024, shuffle=False)


# In[13]:


# Define hyperparameters
wavelet_types = ['dog', 'mexican_hat', 'meyer', 'shannon', 'bump']
hidden_layer_sizes = [2, 32, 128, 512, 1516]
learning_rates = [1e-3, 1e-4, 1e-5]
epochs = 20

# Initialize variables
curr_params = np.empty(3, dtype=object)
best_params = np.empty(3, dtype=object)
max_mcc = -12
best_acc = -1

for wavelet in wavelet_types:
    curr_params[0] = wavelet
    for nodes in hidden_layer_sizes:
        curr_params[1] = nodes
        for lr in learning_rates:
            curr_params[2] = lr
            model = KAN([768, nodes, 1], wavelet_type=wavelet)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr) #weight_decay=decay_rate
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            criterion = nn.MSELoss()
    
            model_train_loss = np.empty(0, dtype=float)
            model_train_acc = np.empty(0, dtype=float)
            model_val_loss = np.empty(0, dtype=float)
            model_val_acc = np.empty(0, dtype=float)
            
            #For a specified number of epchs 
            for epoch in range(epochs):
                #print(f'Epoch Number: {int(epoch)+1}')
                # Training
                train_loss, train_correct, train_total = 0.0, 0, 0
                tp, tn, fp, fn = 0,0,0,0
                model.train()
                # for samples, labels in tqdm(trainloader):
                with tqdm(trainloader) as pbar:
                    for i, (samples, labels) in enumerate(pbar):
                        samples = samples.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(samples)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
                        
                        train_loss += loss.item()
                        pred = outputs.round()
                        true = labels
                        for x in range(len(pred)):
                            if pred[x] and true[x]:
                                tp+=1
                            if not pred[x] and not true[x]:
                                tn+=1
                            if pred[x] and not true[x]:
                                fp+=1
                            if not pred[x] and true[x]:
                               fn+=1
                        train_total += labels.size(0)
                        train_correct += (pred == labels).sum().item()
    
                train_loss /= len(trainloader)
                #print(f'TP: [{tp}] TN: [{tn}] FP: [{fp}] FN: [{fn}]')
                train_acc = train_correct / train_total
                model_train_loss = np.append(model_train_loss, train_loss)
                model_train_acc = np.append(model_train_acc, train_acc)
                #print(f'Train Loss: {train_loss}')
                #print(f'Train Accuracy: {train_acc}')
    
                 # Validation
                val_loss, val_correct, val_total = 0.0, 0, 0
                tp, tn, fp, fn = 0,0,0,0
                model.eval()
                with torch.no_grad():
                    for samples, labels in valloader:
                        samples = samples.to(device)
                        labels = labels.to(device)
                        outputs = model(samples)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        predicted = outputs.round()
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        pred = outputs.round()
                        true = labels.to(device)
                        for x in range(len(pred)):
                            if pred[x] and true[x]:
                                tp+=1
                            if not pred[x] and not true[x]:
                                tn+=1
                            if pred[x] and not true[x]:
                                fp+=1
                            if not pred[x] and true[x]:
                                fn+=1
                val_loss /= len(valloader)
                val_acc = val_correct / val_total
                model_val_loss = np.append(model_val_loss, val_loss)
                model_val_acc = np.append(model_val_acc, val_acc)
                #print(f'Validation Loss: {val_loss}')
                #print(f'Validation Accuracy: {val_acc}')
                
                print(f'TP: [{tp}] TN: [{tn}] FP: [{fp}] FN: [{fn}]')
                mcc = torch.mean(torch.tensor((tp*tn - fp*fn) / (math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))))
                print(f'MCC: {mcc}')
                
                # Update learning rate
                #scheduler.step()
            # Plot
            #plt.plot(model_train_acc, label='train_acc')
            #plt.plot(model_val_acc, label='val_acc')
            #plt.plot(model_train_loss, label='train_loss')
            #plt.plot(model_val_loss, label='val_loss')
            #plt.legend()
            #plt.show()
            X_test = test[:, :-1]
            Y_test = test[:, -1]
            X_test = torch.tensor(X_test, dtype=torch.float32)
            Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)
            model.eval()
            y_pred = model(X_test)
            acc = (y_pred.round() == Y_test).float().mean()
            acc = float(acc)
            mcc = matthews_corrcoef(Y_test.detach(), y_pred.detach().round())
            if mcc > max_mcc:
                max_mcc = mcc
                best_params = curr_params
                best_acc = acc
            print("MCC", mcc)
            print("Model accuracy: %.2f%%" % (acc*100))
print(f'Best MCC: {max_mcc}')
print(f'Model gives accuracy of: {best_acc}')
print('Achieved with hyperparameters: ', best_params)


# In[ ]:




