#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 09:06:38 2021

@author: haider
"""
import os
import sqlite3
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
def NewOldDoms(Events,Eventlist,Nevents):
   "Counts New dom triggers and old dom triggers per event, returns 3-by-Nevents array, first element is old, second is Deggs,third is Mdoms"
   doms=np.zeros((Nevents,3))
   count=0
   for Event_number in Eventlist:
       Event=features.loc[features["event_no"]==Event_number]
       groupdict=Event.groupby("pmt_type").indices
       if 20 in groupdict:
           Old=len(groupdict[20])
           doms[count,0]=Old
       if 120 in groupdict:
           Deggs=len(groupdict[120])
           doms[count,1]=Deggs
       if 130 in groupdict:
           Mdoms=len(groupdict[130])
           doms[count,2]=Mdoms




       count+=1
        

   return doms
def dtdxyz(pulse1,pulse2,i):
    """takes in an event and a pulse, returns two pandas Series, the first is dxyz from that pulse to every other pulse,second is abs(dt),
    includes self element"""
    dx=pulse1['dom_x']-pulse2['dom_x']
    dy=pulse1['dom_y']-pulse2['dom_y']
    dz=pulse1['dom_z']-pulse2['dom_z']
    distances=np.sqrt(+dx**2+dy**2+dz**2)
    distances=distances.drop(distances.index[i])
    dt=abs(pulse1['time']-pulse2['time'])
    dt=dt.drop(dt.index[i])
    return distances,dt
# def SRTclean(Event,dxmax,dtmax):
#     """Takes an event,dxyz and dt, SRT cleans it and returns row number + true or false for if it survived SRT cleaning"""
#     Npulses=len(Event)
#     SRTARR=np.zeros((Npulses,2))
#     for i in range(Npulses):
        
    
#     return SRTARR
datafolder="/home/haider/Master/Data/"
currentset="dev_upgrade_train_step4_001"


distro=datafolder+currentset+"/"+"distributions.pkl"
distributions=open(distro,"rb")
distrodict=pickle.load(distributions)

sets=datafolder+currentset+"/"+"sets.pkl"
setsf=open(sets,"rb")
setsdic=pickle.load(setsf)

transformers=datafolder+currentset+"/"+"transformers.pkl"
transforms=open(transformers,'rb')
transformdict=pickle.load(transforms)


datafilename=datafolder+currentset+"/"+"Data.db"
con=sqlite3.connect(datafilename)

Nevents=2
query_events="select DISTINCT event_no from features LIMIT %d" % (Nevents)
Events =pd.read_sql(query_events,con).squeeze()
c=299792458*10**(-9)

query_truth = 'select * from features WHERE event_no IN %s and SRTInIcePulses = 1 '%(str(tuple(Events)))
truth = pd.read_sql(query_truth, con)
# query_features = 'select * from features WHERE event_no IN %s and SRTInIcePulses = 1 '%(str(tuple(Events)))
query_features = 'select * from features WHERE event_no IN %s'%(str(tuple(Events)))
features = pd.read_sql(query_features, con)

uniques=features["event_no"].value_counts()
maxuni=max(uniques)

"inverse transform features,transform time to microseconds"
for key in transformdict['features']:
    features[key]=transformdict['features'][key].inverse_transform(features[[key]])
features["time"]=features["time"]*1e-3


# "inverse transform truth, transform time to meters"

# for key in transformdict['truth']:
#     truth[key]=transformdict['truth'][key].inverse_transform(truth[[key]])
# truth["time"]= truth["time"]*c
"""SRTcleaning"""
CinIce=224900569e-6 #divided by refractive index, given in microseconds
dtmax=0.5
dxmax=100
Dvmax=dxmax
dtiter=50
dxiter=90
Dvmax+=dxiter
dtmax+=dtiter*0.01
Dvmax=Dvmax
featevent=features[features["event_no"]==155551517]
feateventdxyz,feateventdt=dtdxyz(featevent,featevent.iloc[25],25)
causal=feateventdxyz/feateventdt*(feateventdt<dtmax+0.7)
causalcheck1=(causal!=0)
causalcheck2=causal.loc[(causal!=0)]<Dvmax
if (causal.loc[(causal!=0)]<Dvmax).any():
    print (1)
else:
    print (0)
SRTtrue=featevent["SRTInIcePulses"].to_numpy()
SRTarr=np.zeros(len(featevent))
# for i in range(len(featevent)):
#     feateventdxyz,feateventdt=dtdxyz(featevent,featevent.iloc[i],i)
#     causal=feateventdxyz/feateventdt*(feateventdt<dtmax)
#     if (causal.loc[(causal!=0)]<Dvmax).any():
#         SRTarr[i]=1
    
SRTcomp=SRTarr==SRTtrue
# Iter=100
# Errarr=np.zeros((Iter,Iter))
# SRTtrue=featevent["SRTInIcePulses"].to_numpy()
# for k in range(Iter):
#     Dvmax+=1
#     dtmax=0.5
#     for j in range (Iter):
#         dtmax+=1/Iter
#         SRTarr=np.zeros(len(featevent))
#         for i in range(len(featevent)):
#             feateventdxyz,feateventdt=dtdxyz(featevent,featevent.iloc[i],i)
#             causal=feateventdxyz/feateventdt*(feateventdt<dtmax)
#             if (causal.loc[(causal!=0)]<Dvmax).any():
#                 SRTarr[i]=1
#                 SRTcomp=SRTarr==SRTtrue
#                 Errarr[j,k]=len(SRTcomp)-sum(SRTcomp)
#     print (k)
# dtiter=50
# dxiter=90
# pd.DataFrame(Errarr).to_csv("SRTparamsErrors__upgrade_155551517.csv")
"""plot 2d histogram"""
# Doms=NewOldDoms(features,Events,Nevents)
# NvsO=np.zeros((Nevents,2))
# NvsO[:,0]=Doms[:,0] # new vs old
# NvsO[:,1]=Doms[:,1]+Doms[:,2]
# Totevents=pd.read_csv("results_rasmus.csv")
# heatmap, xedges, yedges = np.histogram2d(NvsO[:,0],NvsO[:,1], bins=20,range=[[0,50],[0,100]])
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# plt.imshow(heatmap.T, extent=extent, origin='lower')
