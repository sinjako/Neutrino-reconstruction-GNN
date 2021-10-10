# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:23:04 2021

@author: Haider
"""

import torch.multiprocessing
import numpy as np
import pickle
import sqlite3
import pandas as pd
import sklearn
import sklearn.metrics
from torch_geometric.data import Data
import torch
import time
import matplotlib.pyplot as plt
def Avgsetdist(Points):
    Npoints=Points.shape[0]
    for i in range(Npoints):
        curr=Points[i]
        currother=np.vstack((Points[:i, :], Points[i+1:,]))
        diff=currother-curr
        dist=np.sqrt(diff[:,0]**2+diff[:,1]**2+diff[:,2]**2)
        avgdist=np.mean(dist)
    return avgdist
if __name__ == '__main__':
    
    datafolder="D:\Schoolwork\Masters\Data\\"
    currentset="dev_numu_train_l5_retro_001\\"
    transformers=datafolder+currentset+"/"+"transformers.pkl"
    transforms=open(transformers,'rb')
    transformdict=pickle.load(transforms)
    datafilename=datafolder+currentset+"Data.db"
    graph_folder="4_nearest_xyzt_alltargets\\"
    targetfolder="D:\Schoolwork\Masters\\Data\\"+currentset+"Graphs\\"+graph_folder
    con=sqlite3.connect(datafilename)
    N_events=500000
    N_events_max=5824118
    N_neighbors=4
    query_events="select DISTINCT event_no from truth LIMIT %s" %(N_events)
    
    Events =pd.read_sql(query_events,con).squeeze()
    # query_events2="select * from truth WHERE event_no IN %s" % (str(tuple(Events)))
    # Events2 =pd.read_sql(query_events2,con).squeeze()
    SRT=1
    
    
    
    Events=Events.to_numpy()
    pulsecount=np.zeros((N_events,3))
    count=0
    
    for event in Events:
        query="select x,y,z from features where event_no=%s" %(event)
        features=pd.read_sql(query,con)
        features=features.to_numpy()
        query="select energy_log10 from truth where event_no=%s" %(event)
        truth=pd.read_sql(query,con)
        truth=truth.to_numpy()
        pulsecount[count,0]=features.shape[0]
        pulsecount[count,1]=truth
        pulsecount[count,2]=Avgsetdist(features)

        
        count+=1
        if count%10000==0:
            print (count)
    # plt.plot(pulsecount[0:count,0],pulsecount[0:count,1])
    
    
    
    # Events=Events.to_numpy()
    # pulsecount=np.zeros((N_events,2))
    # count=0
    # for event in Events:
    #     query="select time from features WHERE event_no = %s and SRTInIcePulses = 1" %(event)
    #     features=pd.read_sql(query,con)
    #     query2="select energy_log10 from truth where event_no=%s" %(event)
    #     energy=pd.read_sql(query2,con)
    #     pulsecount[count,0]=len(features)
    #     pulsecount[count,1]=energy["energy_log10"]
    #     count+=1
    #     if count%10000==0:
    #         print (count)
    # plt.plot(pulsecount[0:count,0],pulsecount[0:count,1])
    # bins=100
    # xbins=np.zeros(bins)
    # ybins=np.zeros(bins)
    # zbins=np.zeros(bins)
    # posspace=np.linspace(-600,600,bins)
    # for event in Events:
    #     query="select x,y,z from features WHERE event_no = %s and SRTInIcePulses = 1" %(event)

    #     features=pd.read_sql(query,con)
    #     features["x"]=transformdict["features"]["x"].inverse_transform(features[["x"]])
    #     features["y"]=transformdict["features"]["y"].inverse_transform(features[["y"]])
    #     features["z"]=transformdict["features"]["z"].inverse_transform(features[["z"]])
        
    #     hitsx=np.searchsorted(posspace,features["x"])
    #     hitsy=np.searchsorted(posspace,features["y"])
    #     hitsz=np.searchsorted(posspace,features["z"])
        
    #     xbins[hitsx]+=1
    #     ybins[hitsy]+=1
    #     zbins[hitsz]+=1

        
    #     count+=1
    #     if count%10000==0:
    #         print (count)

    # plt.figure(0)
    # plt.step(posspace,xbins)
    # plt.figure(1)
    # plt.step(posspace,ybins)
    # plt.figure(2)
    # plt.step(posspace,zbins)
    # pickle.dump(xbins,open("xbins.pkl",'wb'))
    # pickle.dump(ybins,open("ybins.pkl",'wb'))
    # pickle.dump(zbins,open("zbins.pkl",'wb'))
    
    
    