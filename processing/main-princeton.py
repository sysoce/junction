#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:02:57 2019

Electricity consumption for buildings

@author: alirezagoudarzi
"""
import datetime
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_csv('../data/buildings-2018-3.csv')

def getTime(x):
    m=int(x.split(' ')[0].split('/')[0])

    h=int(x.split(' ')[2].split(':')[0])
    if x.split(' ')[3] == 'PM':
        h+=12
    return m,h


energy = data.iloc[:,1].tolist();
times = np.array(data.iloc[:,0].tolist());

days = [ datetime.datetime.strptime(d.split('T')[0],'%Y-%m-%d').day for d in times]
month = [ datetime.datetime.strptime(d.split('T')[0],'%Y-%m-%d').month for d in times]



#clean the NAN
for i in range(len(energy)):
    if np.isnan(energy[i]):
        energy[i] = (energy[i-1] + energy[i+1])/2

#mon_hour = [ [getTime(x)[0],getTime(x)[1]] for x in times ]
winhours=48
inthours=2
plt.plot(energy[0:winhours])
plt.xticks(range(0,len(times[0:winhours]),inthours),times[(range(0,len(times[0:winhours]),inthours))],rotation='vertical')
plt.grid('on')
plt.show()

winL = 4

movAvg = []
for i in range(0,len(energy),winL):
    movAvg.append(sum(energy[i:i+winL])/winL)

plt.plot(movAvg[0:100])