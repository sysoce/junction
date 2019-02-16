# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 21:50:30 2019

@author: alire

Load and process production

https://api.solcast.com.au/pv_power/forecasts?longitude=40.342999&latitude=-74.651333&capacity=1000&api_key=d_wCZ0e8TvX02gjqU3Gcq84jG6SJWUsW&format=json

"""




import json
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

with open('../data/princeton.json', 'r') as f:
    data = json.load(f)

gen = []
times = []

for d in data['forecasts']:
    gen.append(d['pv_estimate'])
    times.append(d['period_end'])

times = np.array(times)
plt.plot(gen)


winhours=len(times)
inthours=2*6

plt.plot(gen[0:winhours])
plt.xticks(range(0,len(times[0:winhours]),inthours),times[(range(0,len(times[0:winhours]),inthours))],rotation='vertical')
plt.grid('on')
plt.show()