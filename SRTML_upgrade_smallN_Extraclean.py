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
from sqlalchemy import create_engine
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
def dtdxyz(pulse1,pulse2,i,Selfclean):
    """takes in an event and a pulse, returns two pandas Series, the first is dxyz from that pulse to every other pulse,second is abs(dt),
    Selfremove=1 doesnt check same unit, selfremove=0 doesnt check same pulse,i is indice of pulse"""
    dx=pulse1['dom_x']-pulse2['dom_x']
    dy=pulse1['dom_y']-pulse2['dom_y']
    dz=pulse1['dom_z']-pulse2['dom_z']
    distances=np.sqrt(+(dx**2)+(dy**2)+(dz**2))
    dt=abs(pulse1['time']-pulse2['time'])
    if Selfclean==1:
            selfcheck=(pulse1["string"] == pulse2["string"]) & (pulse1["dom"] == pulse2["dom"]) & (pulse1["pmt"] == pulse2["pmt"]) # returns series with True if it is not self element
            distances=distances.loc[selfcheck==False]
            dt=dt.loc[selfcheck==False]
    else:
        dt=dt.drop(dt.index[i])
        distances=distances.drop(distances.index[i])# this drops self element, un comment to go back to normal SRT
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
Selfclean=0

featevent=features[features["event_no"]==155551517]
kk=28
#dupes=(featevent["string"] == featevent.loc[kk]["string"]) & (featevent["dom"] == featevent.loc[kk]["dom"]) & (featevent["pmt"] == featevent.loc[kk]["pmt"])
# dupremovd=featevent[["string","dom","pmt"]].drop_duplicates()
# featevent=featevent.loc[dupremovd.index]

feateventdxyz,feateventdt=dtdxyz(featevent,featevent.loc[kk],kk,Selfclean)

causal=feateventdxyz/feateventdt*(feateventdt<dtmax)
causalcheck1=(causal!=0)
causalcheck2=causal.loc[(causal!=0)]<Dvmax
if (causal.loc[(causal!=0)]<Dvmax).any():
    print (1)
else:
    print (0)
SRTtrue=featevent["SRTInIcePulses"]
SRTarr=pd.Series(0,featevent.index,name="SRTInIcePulses")
# for i in range(len(featevent)):
#     feateventdxyz,feateventdt=dtdxyz(featevent,featevent.iloc[i],i,Selfclean)
#     causal=feateventdxyz/feateventdt*(feateventdt<dtmax)
#     if (causal.loc[(causal!=0)]<Dvmax).any():
#         SRTarr.iloc[i]=1
# feateventnew=featevent.copy()
# feateventnew.update(SRTarr)
# SRTcomp=SRTarr==SRTtrue
# print (len(SRTcomp)-sum(SRTcomp))
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
# NeventsTot=780165315
# NeventsToCount=4
# Nevents=2
# TotEvents=pd.Series(dtype=int)
# Ncounted=0
# NvsO=np.zeros((NeventsToCount,2))
# rasmusevent=pd.read_csv("results_rasmus.csv")
# Revents=rasmusevent["event_no"].iloc[0:NeventsToCount]

# while Ncounted<NeventsToCount:
#     # if len(TotEvents)>0:
#     #     query_events="select DISTINCT event_no from features WHERE event_no NOT IN %s LIMIT %d " % (str(tuple(TotEvents)), Nevents)
#     # else:
#     #     query_events="select DISTINCT event_no from features  LIMIT %d " % (Nevents)
#    # Events =pd.read_sql(query_events,con).squeeze()
#     Ncounted+=Nevents
#     Events=Revents.iloc[Ncounted-Nevents:Ncounted]
#     # query_truth = 'select * from truth WHERE event_no IN %s '%(str(tuple(Events)))
#     # truth = pd.read_sql(query_truth, con)
#     query_features = 'select * from features WHERE event_no IN %s '%(str(tuple(Events)))
#     features = pd.read_sql(query_features, con)
#     TotEvents=TotEvents.append(features["event_no"]) #" just changed CHECK IF IT WORKS"
#     "inverse transform features,transform time to meters"
#     for key in transformdict['features']:
#         features[key]=transformdict['features'][key].inverse_transform(features[[key]])
#         features["time"]=features["time"]*c

#     featcopy=features.copy()

#     for Eventnumber in Events:
#         print (Eventnumber)
#         event=features.loc[features["event_no"]==Eventnumber]
#         SRTtrue=event["SRTInIcePulses"]
#         SRTarr=pd.Series(0,event.index,name="SRTInIcePulses")    
#         # dupremoved=event[["string","dom","pmt"]].drop_duplicates()
#         # event=event.loc[dupremoved.index]
#         for i in range(len(event)):

#             eventdxyz,eventdt=dtdxyz(event,event.iloc[i],i,Selfclean)
#             causal=eventdxyz/eventdt*(eventdt<dtmax)
#             if (causal.loc[(causal!=0)]<Dvmax).any():
#                 SRTarr.iloc[i]=1
#         featcopy.update(SRTarr)
#     # Doms=NewOldDoms(features.loc[features["SRTInIcePulses"]==1],Events,Nevents)
#     Doms=NewOldDoms(featcopy.loc[featcopy["SRTInIcePulses"]==1],Events,Nevents)
#     NvsO[Ncounted-Nevents:Ncounted,0]=Doms[:,0]
#     NvsO[Ncounted-Nevents:Ncounted,1]=Doms[:,1]+Doms[:,2]
#     if Ncounted%Nevents==0:
#         print (Ncounted)


# heatmap, xedges, yedges = np.histogram2d(NvsO[:,0],NvsO[:,1], bins=20,range=[[0,50],[0,200]])
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# plt.imshow(heatmap.T, extent=extent, origin='lower',aspect=0.1)

# featcopy.to_csv("HaiderSRTevents.csv")
# featcopy.to_csv("HaiderSRTeventsNoDup.csv")
# print (len(featcopy.loc[featcopy["SRTInIcePulses"]==1]))
# print (len(features.loc[features["SRTInIcePulses"]==1]))