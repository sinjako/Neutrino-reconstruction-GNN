# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import sqlite3
import pickle
import lightgbm as lgb

def propdist(pulse1,pulse2):
    "return proper distance in flat space"
    dx=pulse1['x']-pulse2['x']
    dy=pulse1['y']-pulse2['y']
    dz=pulse1['z']-pulse2['z']
    dt=pulse1['time']-pulse2['time']
    return np.sqrt(+dx**2+dy**2+dz**2+dt**2)

"initial operations"

c=299792458*10**(-6) # speed of light, multiplied by 1e-6 since i believe data is given in nanoseconds


transforms=open("transformers.pkl",'rb')
transformdict=pickle.load(transforms)
con=sqlite3.connect("dev_numu_train_l5_retro_001.db")


Nevents=10000
query_events="select DISTINCT event_no from features LIMIT %d" % (Nevents)
Events =pd.read_sql(query_events,con).squeeze()



query_truth = 'select * from truth WHERE event_no IN %s '%(str(tuple(Events)))
truth = pd.read_sql(query_truth, con)
query_features = 'select * from features WHERE event_no IN %s '%(str(tuple(Events)))
features = pd.read_sql(query_features, con)


"inverse transform features,time converted with speed of light"
for key in transformdict['features']:
    features[key]=transformdict['features'][key].inverse_transform(features[[key]])
features['time']=features['time']
times=np.unique(features["time"])
print (times.shape)
uniques=features["event_no"].value_counts()
print (max(uniques))

# "inverse transform truth, time converted with speed of light"

# for key in transformdict['truth']:
#     truth[key]=transformdict['truth'][key].inverse_transform(truth[[key]])
# truth['time']=truth['time']*c
# "Calculate closest neighbors in proper distance"
# N=20 # Neighbors
# Npulses=features.shape[0]
# Neighbors=np.zeros((Npulses,2,N))
# for i in range(Npulses):
#     Distances=propdist(features,features.iloc[i]).drop(i)
#     NearIndex=Distances.nsmallest(N)   #Nearindex[i] gives the corresponding element.
#     Neighbors[i,0,:]=NearIndex.index
#     Neighbors[i,1,:]=NearIndex
    
# "Create target array of SRT info, create SRT features by inserting relevant figures"
# target=features["SRTInIcePulses"]

# SRTdist=pd.DataFrame(Neighbors[:,1,:])
# SRTdist["charge_log10"]=features["charge_log10"]
# SRTdist["charge_log10"]=features["charge_log10"]