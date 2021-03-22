#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 05:16:41 2021

@author: haider
"""
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
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj
from tqdm import tqdm
from scipy.spatial import distance_matrix
def Icedist(Data):
    set1=set(features["dom"])
    set2=set(features['string'])
    posdata=[]
    columns=["string","dom","x","y","z"]
    for i in set1:
        for j in set2:
            detectors=features.loc[(features["string"]==j) & (features["dom"]==i)]
            if detectors.empty==0:
                posdata.append([j,i,detectors.iloc[0]["x"],detectors.iloc[0]["y"],detectors.iloc[0]["z"]])
    pos=pd.DataFrame(data=posdata,columns=columns)
    posnp=np.zeros((len(pos),3))
    posnp[:,0]=pos["x"]
    posnp[:,1]=pos["y"]
    posnp[:,2]=pos["z"]
    distmatrix=distance_matrix(posnp,posnp)
    return distmatrix,set1,set2,pos

"PROBLEM: Adjacency output too little, probably all fucked"
def Edgefunction(N_Neighbors,data):
    n=N_Neighbors
    distmatrix,set1,set2,pos=Icedist(data)
    Adjmatrix=np.zeros((len(data),len(data)))
    count=0
    for i in set1:
        for j in set2:
            if pos.loc[(pos["dom"]==i) & (pos["string"]==j)].empty==0:
                idx=np.argsort(distmatrix[count,:][0:n+1])
                count+=1
                count2=0
                for k in idx:
                    if count2==0:
                        count2+=1
                        nodestring=pos.iloc[k]["string"]
                        nodedom=pos.iloc[k]["dom"]
                        string=pos.iloc[k]["string"]
                        dom=pos.iloc[k]["dom"]
                        nodeind=features.loc[(features["string"]==nodestring) & (features["dom"]==nodedom)].index
                        neighborind=features.loc[(features["string"]==string) & (features["dom"]==dom)].index
                        for o in nodeind:
                            Adjmatrix[nodeind,neighborind]=1
                            print (nodeind,neighborind,Adjmatrix[nodeind,neighborind],"\n")
    return Adjmatrix,distmatrix
class IceGraph(Dataset):
    
    def __init__(self, data,features,truth,labels,**kwargs):
        """
        Create spektral graph dataset from icecube Data. data is a pandas DataFrame of pulses,
        features is a dictionary of features,
        truth is a pandas Dataframe to train on, labels is a dictionary of the features from truth to be trained on,
        **kwargs is there to pass argument to ther parent class Dataset
        """
        self.N_neighbors=4

        super().__init__()
    def Edgefunction(N_Neighbors,data,self):
        n=N_Neighbors
        distmatrix,set1,set2,pos=Icedist(data)
        Adjmatrix=np.zeros((self.n_nodes,self.n_nodes))
        count=0
        for i in set1:
            for j in set2:
                idx=np.argsort(distmatrix[count,:][0:n+1])
                count+=1
                count2=0
                for k in idx:
                    if count2==0:
                        count2+=1
                        nodestring=pos.iloc[k]["string"]
                        nodedom=pos.iloc[k]["dom"]
                    string=pos.iloc[k]["string"]
                    dom=pos.iloc[k]["dom"]
                    nodeind=features.loc[(features["string"]==nodestring) & (features["dom"]==nodedom)].index
                    neighborind=features.loc[(features["string"]==string) & (features["dom"]==dom)].index
                    for o in nodeind:
                        Adjmatrix[nodeind,neighborind]=1
                    
                
                
    def read(self):
        return 
        
"initial operations"

c=299792458*10**(-6) # speed of light, multiplied by 1e-6 since i believe data is given in nanoseconds


transforms=open("transformers.pkl",'rb')
transformdict=pickle.load(transforms)
con=sqlite3.connect("dev_numu_train_l5_retro_001.db")


Nevents=200
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
    
Adj=Edgefunction(4,features)