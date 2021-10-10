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

datafolder="/groups/hep/haider/Masters/Data/"
currentset="dev_numu_train_l5_retro_001/"
datafilename=datafolder+currentset+"Data.db"
graph_folder="4_nearest_xyzt_alltargets\\"
targetfolder="D:\Schoolwork\Masters\\Data\\"+currentset+"Graphs\\"+graph_folder
con=sqlite3.connect(datafilename)
feat_list="x,y,z,time"
target_list="energy_log10,time,vertex_x,vertex_y,vertex_z,direction_x,direction_y,direction_z,azimuth,zenith,pid,interaction_type"
N_events=10
N_events_max=5824118
N_neighbors=4
query_events="SELECT * from truth LIMIT %s " %(N_events)
query_events2= "SELECT *from features LIMIT %s" %(N_events)
print (pd.read_sql(query_events, con))
print (pd.read_sql(query_events2, con))