# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 03:44:31 2018

@author: rajes
"""
import numpy as np

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size  = 2
hidden_size = 2
num_classes = 4

num_epochs  = 5
batch_size  = 100
learn_rate  = .001


class QPSKDataset(Dataset):
    def __init__(self,c):
        X = np.loadtxt(c, delimiter = ',', dtype = np.float32)
        X1 = X[:,0:-1]
        X2 = X[:,[-1]]
        X2 = X2.astype(np.int64)
        self.len    = X.shape[0]
        self.x_data = torch.from_numpy(X1)
        self.y_data = torch.from_numpy(X2)
        

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

train_dataset = QPSKDataset('Train.csv')

test_dataset  = QPSKDataset('Test.csv')


train_loader = DataLoader(dataset    = train_dataset, 
                          batch_size = 100,
                          shuffle    = True)


test_loader = DataLoader(dataset    = test_dataset, 
                          batch_size = 100,
                          shuffle    = True)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)

# Train the model

total_step = len(train_loader)

for epoch in range(num_epochs):
    for i,data in enumerate(train_loader,0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        
        # Forward Pass
        output = model(inputs)
        loss   = criterion(output,torch.max(labels, 1)[0])
        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if((i+1)%1000 == 0):
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

