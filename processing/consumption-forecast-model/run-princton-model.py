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



BLDNUM = 1

print('start loading the data from disk')
# don't get the rows with none entries in the first file
data = pd.read_csv('../../data/buildings-2018-'+str(2)+'.csv')
energy = data.iloc[402:,BLDNUM].tolist();
times = (data.iloc[402:,0].tolist());
mdh = [ getTime(t) for t in times]

# load the rest of files and add to dataset
for i in range(3,12):
    data = pd.read_csv('../../data/buildings-2018-'+str(i)+'.csv')
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



print('setting up the data sources')
batch_size = 32
maxEpoch = 10
hidden_size = 100
num_h_layers = 1



print('setting up the model and the trainer')
# model  = ConsumCastV2(1,hidden_size,num_h_layers)
model  = ConsumCastV1(1,hidden_size,num_h_layers)

print('choosing building and loading the model')

states = np.load('model_'+type(model).__name__+'-'+data.columns[BLDNUM]+'.pt.npy')
model.load_state_dict(states.item())

model.training=True


print('validate')
model.training=False

#get sample input normalize
x = Variable(torch.from_numpy(np.array([all_x[6][:50]])/normalize).unsqueeze(2)).float()
y = Variable(torch.from_numpy(np.array([all_y[6]])/normalize)).float()

mdh = Variable(torch.from_numpy(np.array([all_mdh[50-1][:]])/normalize)).float()


# run the model
out = model(x,mdh)


plt.plot(out.data[0].numpy(),label='forecast')
plt.plot(y.data[0].numpy(),label='ground truth')
plt.legend()
plt.show()

