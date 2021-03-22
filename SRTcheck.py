#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 18:09:38 2021

@author: haider
"""

import os
import sqlite3
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

datafolder="/home/haider/Master/Data/"
currentset="dev_upgrade_train_step4_001"
datafilename=datafolder+currentset+"/"+"Data.db"
con=sqlite3.connect(datafilename)


Events=pd.read_csv("HaiderSRTeventsNoDup.csv",index_col=0)
Eventarr=pd.unique(Events["event_no"])
Eventsub=Eventarr[0:1000]

totsum=0
count=0
for Event_number in Eventsub:
    count+=1
    if count%100==0:
        print (count)
    Event=Events.loc[Events["event_no"]==Event_number]
    eventsum=Event["SRTInIcePulses"].sum()
    if eventsum>0:
        totsum+=1
print (totsum)
