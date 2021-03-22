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
import matplotlib.pyplot as plt
def propdist(pulse1,pulse2):
    "return proper distance in flat space"
    dx=pulse1['x']-pulse2['x']
    dy=pulse1['y']-pulse2['y']
    dz=pulse1['z']-pulse2['z']
    dt=pulse1['time']-pulse2['time']
    return np.sqrt(+dx**2+dy**2+dz**2+dt**2)

"initial operations"

c=299792458*10**(-9) # speed of light, multiplied by 1e-9 since i believe data is given in nanoseconds


transforms=open("transformers.pkl",'rb')
transformdict=pickle.load(transforms)
con=sqlite3.connect("dev_numu_train_l5_retro_001.db")


Nevents=2
query_events="select DISTINCT event_no from features LIMIT %d" % (Nevents)
Events =pd.read_sql(query_events,con).squeeze()



query_truth = 'select * from truth WHERE event_no IN %s '%(str(tuple(Events)))
truth = pd.read_sql(query_truth, con)
query_features = 'select * from features WHERE event_no IN %s '%(str(tuple(Events)))
features = pd.read_sql(query_features, con)
uniques=features["event_no"].value_counts()
maxuni=max(uniques)

"inverse transform features,transform time to meters"
for key in transformdict['features']:
    features[key]=transformdict['features'][key].inverse_transform(features[[key]])
features["time"]=features["time"]*c



"inverse transform truth, transform time to meters"

for key in transformdict['truth']:
    truth[key]=transformdict['truth'][key].inverse_transform(truth[[key]])
truth["time"]= truth["time"]*c
    
"Calculate closest neighbors in proper distance"
N=20 # Neighbors
Npulses=features.shape[0]
Columns=["charge_log10","Distance"]
Neighborvals=np.zeros((Npulses,N*len(Columns)))
"Find and create neighbors"
for i in (Events):
    Eventpulses=features.loc[features["event_no"]==i]
    Eventsize=len(Eventpulses.index)
    for j in range(Eventsize):
        Distances=propdist(Eventpulses, Eventpulses.iloc[j])
        Distances=Distances.drop(Distances.index[j])
        Nearby=Distances.nsmallest(N)
        Nearevents=Eventpulses.loc[Nearby.index]
        Nearevents.insert(14,"Distance",Nearby)
        Nneighbors=len(Nearby.index)
        CurrentPulse=Eventpulses.index[j]
        colnum=0
        for col in Columns:
            Neighborvals[CurrentPulse,colnum*Nneighbors:(colnum+1)*Nneighbors]=Nearevents[col]
            colnum=colnum+1
        
# "Create target array of SRT info from features, initialize lightgbm and train"
# target=np.array(features["SRTInIcePulses"])
# split=50000
# x_train=Neighborvals[split:,:]
# y_train=target[split:]
# x_validate=Neighborvals[:split,:]
# y_validate=target[:split]
# params={"num_leaves":31,"objective":"binary","num_iterations":1000}

# train_data=lgb.Dataset(x_train,label=y_train)
# model=lgb.train(params,train_data)


# "plot results from model"
# Nbins=40
# plt.figure(0)
# ypred=model.predict(x_train)
# resid=abs(y_train-ypred)
# plt.hist(resid,Nbins)
# plt.figure(1)
# ypred2=model.predict(x_validate)
# resid2=abs(y_validate-ypred2)
# plt.hist(resid2,Nbins)
# plt.figure(2)
# plt.hist(ypred2[np.where(ypred2>0)],Nbins)
# yindreal=np.where(y_validate>0.9)
# yindpred=np.where(ypred2>0.5)


# "loop for training"

# Niter=100
# stdarray=np.zeros(100)
# countdiff=np.zeros(100)
# for i in range(Niter):
#     print (i)
#     params={"num_leaves":31,"objective":"binary","num_iterations":(i+1)*10}
#     model=lgb.train(params,train_data)
#     modpred=model.predict(x_validate)
#     stdarray[i]=np.std(abs(modpred-y_validate))
#     countdiff[i]=len(yindreal[0])-len(np.where(modpred>0.8)[0])
    