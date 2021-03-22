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
import time
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

db_file = 'dev_numu_train_l5_retro_001.db'
start=time.time()
### EXTRACT A LIST OF EVENT_NO's FOR ALL EVENTS IN THE DATABASE
with sqlite3.connect(db_file) as con:
    query = 'select * from truth'
    truth = pd.read_sql(query, con)
events = truth.loc[:,'event_no']  ## THESE ARE OUR KEYS
events  = events[0:10]            ## FIRST TEN OF ALL KEYS
## PULL OUT TRUTH VALUES AND EVENT DATA
with sqlite3.connect(db_file) as con:
    query = 'select * from truth where event_no IN %s '%(str(tuple(events)))
    truth = pd.read_sql(query, con)
    query = 'select * from features WHERE event_no IN %s '%(str(tuple(events)))
    features = pd.read_sql(query, con)
    
    
end=time.time()
print(start-end)