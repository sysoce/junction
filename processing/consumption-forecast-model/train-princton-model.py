#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:02:57 2019

@author: alirezagoudarzi
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as td
import torch.optim as optim

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

from model import * 



def getTime(x): 
    d = datetime.datetime.strptime(x[:-1],'%Y-%m-%dT%H:%M:%S')
    return [d.month/12, d.day/31, d.hour/23]

# Setting the building number to model
print('setting building number to model')
BLDNUM = 1


print('start loading the data from disk')
# don't get the rows with none entries in the first file
data = pd.read_csv('../data/princton/buildings-2018-'+str(2)+'.csv')
energy = data.iloc[402:,BLDNUM].tolist();
times = (data.iloc[402:,0].tolist());
mdh = [ getTime(t) for t in times]

# load the rest of files and add to dataset
for i in range(3,12): 
    data = pd.read_csv('../data/princton/buildings-2018-'+str(i)+'.csv')
    energy += data.iloc[:,BLDNUM].tolist();
    times += (data.iloc[:,0].tolist());
    mdh += [ getTime(t) for t in times]


#clean the NAN
for i in range(len(energy)): 
    energy[i]=float(energy[i])
    if np.isnan(energy[i]): 
        energy[i] = (energy[i-1] + energy[i+1])/2

print('preprocessing the data')
winhours=48
inthours=2

inputLen=50
predictionLen = 24
normalize=100
all_x = [] 
all_y = []
all_mdh = []

for i in range(len(energy)-inputLen-predictionLen):
    all_x.append(energy[i:i+inputLen])
    all_y.append(energy[i+inputLen:i+predictionLen+inputLen])
    all_mdh.append(mdh[i+inputLen-1])

#divide train and validation
train_ratio=.7
idx = list(range(len(all_x)))
np.random.shuffle(idx)
train_idx = idx[:int(len(all_x)*train_ratio)]
val_idx = idx[int(len(all_x)*train_ratio):]
    
train_x = torch.from_numpy(np.array(all_x)[train_idx]/normalize).float().unsqueeze(2)
train_y = torch.from_numpy(np.array(all_y)[train_idx]/normalize).float()
train_mdh = torch.from_numpy(np.array(all_mdh)[train_idx]).float()

val_x = torch.from_numpy(np.array(all_x)[val_idx]/normalize).float().unsqueeze(2)
val_y = torch.from_numpy(np.array(all_y)[val_idx]/normalize).float()
val_mdh = torch.from_numpy(np.array(all_mdh)[val_idx]).float()


print('setting up the data sources')
batch_size = 32
maxEpoch = 10
hidden_size = 100
num_h_layers = 1

train_data = td.TensorDataset(train_x,train_mdh,train_y)
val_data = td.TensorDataset(val_x,val_mdh,val_y)

train_loader = td.DataLoader(train_data,batch_size=batch_size,shuffle=True)
val_loader = td.DataLoader(val_data,batch_size=batch_size,shuffle=True)


print('setting up the model and the trainer')
model  = ConsumCastV1(1,hidden_size,num_h_layers)
opt= optim.Adam(model.parameters(),lr=1e-2)

model.training=True

print('start training')
for e in range(maxEpoch):
   print('starting epoch {:d}'.format(e))
   o,y,L= train_step(opt,model,train_loader,5)

opt= optim.Adam(model.parameters(),lr=1e-3)

for e in range(maxEpoch):
   print('starting epoch {:d}'.format(e))
   o,y,L= train_step(opt,model,train_loader,5)


print('validate')
model.training=False
o,y,L= train_step(opt,model,val_loader,5,train=False)

plt.plot(o.data[0].numpy(),label='forecast')
plt.plot(y.data[0].numpy(),label='ground truth')
plt.legend()
plt.show()

np.save('model_'+type(model).__name__+'-'+data.columns[BLDNUM]+'.pt',model.state_dict())