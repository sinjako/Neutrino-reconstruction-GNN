#!/usr/bin/env python3

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 07:09:19 2021

@author: haider
"""
import os
import pandas as pd
import numpy as np
import sqlite3
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import CategoricalCrossentropy,MeanSquaredError
from tensorflow.keras.metrics import categorical_accuracy,MSE
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj
from tqdm import tqdm
from spektral.models import GeneralGNN

def Icetime(pulse1,pulse2):
    return pulse1["time"]-pulse2["time"]
def Icedist(pulse1,pulse2):
    "Returns distance between two pulses. if pulse1 is given as multiple pulses, it returns all distances between pulse1 and pulse2."
    dx=pulse1['x']-pulse2['x']
    dy=pulse1['y']-pulse2['y']
    dz=pulse1['z']-pulse2['z']
    return np.sqrt(+dx**2+dy**2+dz**2)

class IceGraph(Dataset):
    
    def __init__(self, data,features,truth,labels,Events,funclist,edgedist,**kwargs):
        """
        Create spektral graph dataset from icecube Data. data is a pandas DataFrame of pulses,
        features is a dictionary of features,
        truth is a pandas Dataframe to train on, labels is a dictionary of the features from truth to be trained on,Events is the Series of Events,
        funclist is a list of functions that take two pulses and gives a scalar or categorical edge feature,
        edgedist is a value to be set to 1 to have distance as edge feature
        **kwargs is there to pass argument to ther parent class Dataset
        """
        self.N_neighbors=4
        self.Events=Events
        self.data=data
        self.features=features
        self.labels=labels
        self.truth=truth[self.labels]
        self.funclist=funclist
        self.edgedist=edgedist
        super().__init__(**kwargs)

                
    def read(self):
            def make_graph(event,eventno):
                Npulse=len(event)
                x=np.array(event[self.features])
                a=np.zeros((Npulse,Npulse))
                if self.edgedist==1 or len(self.funclist)>0:
                    e=np.zeros((Npulse,Npulse,self.edgedist+len(self.funclist)))
                y=self.truth.loc[self.truth["event_no"]==eventno].drop(columns="event_no")
                idxstart=event.index[0] # subtract from indices to get corresponding element in a
                for i in range(Npulse):
                    "loop that generates the events adjacency matrix, this matrix is not necessarily symmetrical since your neighbor may have someone closer"
                    distances=Icedist(event,event.iloc[i]).nsmallest(self.N_neighbors+1)#gets indices in event, keep selfelement so the diagonal is preserved
                    Distidx=distances.index -idxstart #subtract idxstart to be able to put into a
                    #print (Distidx,Npulse,i)
                    a[i,Distidx]=1

                    if self.edgedist==1:
                        e[i,Distidx,0]=distances
                        if len(self.funclist)>0:
                            edgefeatcount=1
                            for func in self.funclist:
                                e[i,Distidx,edgefeatcount]=func(event.iloc[Distidx],event.iloc[i])
                if self.edgedist==1 or len(self.funclist)>0:
                    return Graph(x=x, a=sp.csr_matrix(a), y=y,e=e)
                else:
                    return Graph(x=x, a=sp.csr_matrix(a), y=y)
            return [make_graph(self.data.loc[self.data["event_no"]==eventno],eventno) for eventno in self.Events ]
    def datadel(self):
        """delete unnecessary data after making graph"""
        del self.N_neighbors
        del self.Events
        del self.data
        del self.features
        del self.labels
        del self.truth
"initial operations"

c=299792458*10**(-9) # speed of light, multiplied by 1e-6 since i believe data is given in nanoseconds


transforms=open("transformers.pkl",'rb')
transformdict=pickle.load(transforms)
con=sqlite3.connect("dev_numu_train_l5_retro_001.db")


Nevents=100
query_events="select DISTINCT event_no from features LIMIT %d" % (Nevents)
Events =pd.read_sql(query_events,con).squeeze()



query_truth = 'select * from truth WHERE event_no IN %s '%(str(tuple(Events)))
truth = pd.read_sql(query_truth, con)
query_features = 'select * from features WHERE event_no IN %s '%(str(tuple(Events)))
features = pd.read_sql(query_features, con)
uniques=features["event_no"].value_counts()
maxuni=max(uniques)

"inverse transform features"
for key in transformdict['features']:
    features[key]=transformdict['features'][key].inverse_transform(features[[key]])

"inverse transform truth"

for key in transformdict['truth']:
    truth[key]=transformdict['truth'][key].inverse_transform(truth[[key]])

"initialization of graphs and GNN"
targets=["energy_log10","event_no"] # need to pass event_no here, it gets removed later
params=["x","y","z","time","charge_log10"]
edgeswithdistance=1 # set to 1 to include distances in edges
Functions=[]
data=IceGraph(features,params,truth,targets,Events,Functions,edgeswithdistance)
data.datadel()





"""GNN initial values"""
#Parameters
batch_size = 5
learning_rate = 0.01
epochs = 10

F=data.n_node_features
S=data.n_edge_features
"dataset split"
np.random.shuffle(data)
split = int(0.8 * len(data))
data_tr, data_te = data[:split], data[split:]

loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_te = DisjointLoader(data_te, batch_size=batch_size)

"creat model"
#Create model
model = GeneralGNN(data.n_labels, activation="linear")
optimizer = Adam(learning_rate)
loss_fn = MeanSquaredError()

