#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 21:26:16 2019

@author: alirezagoudarzi
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# this model only looks at the timeseries
class ConsumCastV1(nn.Module):
    def __init__(self,i0,h0,l0):
        super(ConsumCastV1, self).__init__()
        
        self.features = nn.GRU(i0,h0,l0,batch_first=True)
        
        self.readout = nn.Sequential(nn.Linear(h0,200),
                                     nn.ReLU(),
                                     nn.Linear(200,24))
 
    def forward(self,x,mdh):
        out = self.features(x);
        out = F.relu(out[0][:,-1,:])
        out = self.readout(out)
        
        return out;

# this model looks at time series and month, day, hour
class ConsumCastV2(nn.Module):
    def __init__(self,i0,h0,l0):
        super(ConsumCastV2, self).__init__()
        
        self.features = nn.GRU(i0,h0,l0,batch_first=True)
        
        self.readout = nn.Sequential(nn.Linear(h0+3,200),
                                     nn.Tanh(),
                                     nn.Linear(200,24))
 
    def forward(self,x,mdh):
        out = self.features(x);
        out = F.tanh(out[0][:,-1,:])
        out = torch.cat((out,mdh),dim=1)

        out = F.relu(self.readout(out))
        
        return out;
    
def train_step(optimizer,net,loader,interval=10,train=True): 
    
    L = []
    for idx, (x,mdh,y) in enumerate(loader): 
        x, mdh, y = Variable(x), Variable(mdh), Variable(y)
        if train:
          optimizer.zero_grad()
        
        out = net(x,mdh)
        loss = F.mse_loss(out,y)
        if train:
          loss.backward()
          optimizer.step()
        
        if idx%interval==0 : 
            print('batch {:d}, loss: {:5.4f}'.format(idx,loss.data.item()))
        L.append(loss.data.item())
    return out,y,L

