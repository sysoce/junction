from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

import mysql.connector

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
import os
from predictor.model import * 
import json

mydb = mysql.connector.connect(
  host="localhost",
  port="3306",
  user="root",
  passwd="",
  database="gridless"
)

def index(request):
    res = []
    print(os.getcwd())

    if request: 
        res.append("Request received" )
    if mydb: 
        res.append("MYSQL Connection established")
    res.append("Hello, world, predictor.")

    mycursor = mydb.cursor()
    mycursor.execute("SELECT count(*) FROM princeton")
    myresult = mycursor.fetchall()
    for x in myresult:
        print(x)

    print('*** NOW THE MODEL ***')




    def getTime(x): 
        d = datetime.datetime.strptime(x[:-1],'%Y-%m-%dT%H:%M:%S')
        return [d.month/12, d.day/31, d.hour/23]



    # Setting the building number to model
    print('setting building number to model')
    BLDNUM = 1


    print('start loading the data from disk')
    # don't get the rows with none entries in the first file
    data = pd.read_csv('../../data/princton/buildings-2018-'+str(2)+'.csv')
    energy = data.iloc[402:,BLDNUM].tolist();
    times = (data.iloc[402:,0].tolist());
    mdh = [ getTime(t) for t in times]

    # load the rest of files and add to dataset
    for i in range(3,12): 
        data = pd.read_csv('../../data/princton/buildings-2018-'+str(i)+'.csv')
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
    model  = ConsumCastV2(1,hidden_size,num_h_layers)

    print('choosing building and loading the model')
    BLDNUM = 1
    states = np.load('../../code/model_'+type(model).__name__+'-'+data.columns[BLDNUM]+'.pt.npy')
    model.load_state_dict(states.item())

    model.training=True


    print('validate')
    model.training=False

    #get sample input normalize
    x = Variable(torch.from_numpy(np.array([all_x[6][:50]])/normalize).unsqueeze(2)).float()
    y = Variable(torch.from_numpy(np.array([all_y[6]])/normalize)).float()

    mdh = Variable(torch.from_numpy(np.array([all_mdh[50-1][:]])/normalize)).float()


    # run the model
    out = model(x,mdh)*normalize
    pred = {}
    for idx,d in enumerate(out.data.numpy()[0]): 
        pred[idx] = str(d)
    print(pred)
    print("*** now return result ***")
    return HttpResponse(json.dumps(pred))
