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
       if 120 in groupdict:
           Deggs=len(groupdict[120])
       if 130 in groupdict:
           Mdoms=len(groupdict[130])

       doms[count,0]=Old
       doms[count,1]=Deggs
       doms[count,2]=Mdoms
       count+=1
        

   return doms
def Domtriggers(Events,Eventlist,Nevents):
    "Return marginal distribution of dom triggers per event"
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

#query_count="SELECT count(*) from features"
#Totalcount=pd.read_sql(query_count,con)
NeventsTot=780165315
NeventsToCount=1000000
Nevents=2
query_events="select DISTINCT event_no from features LIMIT %d" % (Nevents)
Events =pd.read_sql(query_events,con).squeeze()
c=299792458*10**(-9)


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

SRTparams=(150,1E-6)
Doms=NewOldDoms(features,Events,Nevents)


heatmap, xedges, yedges = np.histogram2d(NvsO[:,0],NvsO[:,1], bins=50,range=[[0,200],[0,200]])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.imshow(heatmap.T, extent=extent, origin='lower')

