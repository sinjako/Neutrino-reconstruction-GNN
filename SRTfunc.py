#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:17:45 2021

@author: haider
"""

import os
import sqlite3
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
import time
def dtdxyz(Event,Pulse):
    """takes in an event and a pulse, returns distance between the pulse and all other pulses in event and difference in time
    expects Event to be Nx4 array and Pulse to be 4, ordered as x y z time."""
    dxyz=np.sqrt((Event[:,0]-Pulse[0])**2 +(Event[:,1]-Pulse[1])**2 +(Event[:,2]-Pulse[2])**2)
    dt=abs(Event[:,3]-Pulse[3])
    return dxyz,dt
def SRTclean(Event,dtmax,Dvmax,SelfClean):
    """SRT cleans one event, selfclean=1 stops it from triggering on own unit, Dupliremove removes all but the first pulse from one unit
    expects the pandas dataframe to have only the following columns in the following order
    event_no,string,dom,pmt,dom_x,dom_y,dom_z,time,pmt_type,SRTInIcePulses"""
    EventArr=Event.to_numpy()
    Npulses=EventArr.shape[0]
    for i in range(Npulses):
        dxyz,dt=dtdxyz(EventArr[:,4:8],EventArr[i,4:8])

        if SelfClean==1:
            sameidx=np.where((EventArr[:,1]==EventArr[i,1]).astype(int)+(EventArr[:,2]==EventArr[i,2]).astype(int)+(EventArr[:,3]==EventArr[i,3]).astype(int)==3)
            dt[sameidx]=10000
            causal=dxyz/dt*(dt<dtmax)
            idx=np.where(causal!=0)
            if sum(causal[idx]<Dvmax)>0:
                EventArr[i:,9]=1
            else:
               EventArr[i:,9]=0
        else:
            dt[i]=100000
            causal=dxyz/dt*(dt<dtmax)
            idx=np.where(causal!=0)
            if sum(causal[idx]<Dvmax)>0:
                 EventArr[i:,9]=1
            else:
                EventArr[i:,9]=0
    return EventArr[:,[0,9]].astype(int)
def SRTcleanJob(Events,Eventlist):
    dtmax=1
    dxmax=150
    Dvmax=dxmax
    SelfClean=1
    Dropdup=1
    Length=len(Events)
    Dataout=np.zeros((Length,2))

    for event_number in Eventlist:
        event=Events.loc[Events["event_no"]==event_number]
        if Dropdup==1:
            event=event.loc[(event[["string","dom","pmt"]].drop_duplicates()).index]
        indx=event.index
        print (indx)
        Dataout[indx,:]=SRTclean(event,dtmax,Dvmax,SelfClean)
    Dataout=pd.DataFrame(data=Dataout,index=Events.index,columns=["event_no","SRTInIcePulses"])
    return Dataout
datafolder="/home/haider/Master/Data/"
currentset="dev_upgrade_train_step4_001"

transformers=datafolder+currentset+"/"+"transformers.pkl"
transforms=open(transformers,'rb')
transformdict=pickle.load(transforms)

datafilename=datafolder+currentset+"/"+"Data.db"
con=sqlite3.connect(datafilename)



Nevents=2
query_events="select DISTINCT event_no from features LIMIT %d" % (Nevents)
Events =pd.read_sql(query_events,con).squeeze()
c=299792458*10**(-9)


# query_features = 'select * from features WHERE event_no IN %s and SRTInIcePulses = 1 '%(str(tuple(Events)))
query_features = 'SELECT event_no,string,dom,pmt,dom_x,dom_y,dom_z,time,pmt_type,SRTInIcePulses from features WHERE event_no IN %s'%(str(tuple(Events)))
features = pd.read_sql(query_features, con)

uniques=features["event_no"].value_counts()
maxuni=max(uniques)

"inverse transform features,transform time to microseconds"
for key in transformdict['features']:
    if key in features.columns:
        features[key]=transformdict['features'][key].inverse_transform(features[[key]])
features["time"]=features["time"]*1e-3
dataparams="event_no,string,dom,pmt,dom_x,dom_y,dom_z,time,pmt_type,SRTInIcePulses" # just to remind myself


CinIce=224900569e-6 #divided by refractive index, given in microseconds

featevent=features[features["event_no"]==155551517]
kk=28
featevent2=featevent.to_numpy()



# dupremovd=featevent[["string","dom","pmt"]].drop_duplicates()
# featevent=featevent.loc[dupremovd.index]
dtmax=1
dxmax=150
Dvmax=dxmax
SelfClean=1



feateventdxyz,feateventdt=dtdxyz(featevent2[:,4:8],featevent2[kk,4:8])
feateventdt[kk]=10000
causal=feateventdxyz/feateventdt*(feateventdt<dtmax)
idx=np.where(causal!=0)
if sum(causal[idx]<Dvmax)>0:
    print (1)
start=time.time()
Featevent2SRT=SRTclean(featevent,dtmax,Dvmax,SelfClean)
end=time.time()
print (start-end)
SRTcomp=Featevent2SRT[:,1]==featevent2[:,9]
Mystuff=SRTcleanJob(features,Events)