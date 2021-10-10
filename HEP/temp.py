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
def NearestNeighborEdges(features,N_neighbors):
    """ takes in numpy array where the first 3 columns are xyz, returns list of edges of N nearest neighbors"""
    dist=sklearn.metrics.pairwise_distances(features[:,0:3]) # change these indices if features pulled out change
    edge_index=list()
    for i in range(features.shape[0]):
        nodearr=np.delete(dist[:,i],i) # when deleting, objects higher in indice fall in indice by 1, ie slicing at element 20 makes every element higher than it in number less than 1 so element 21 becomes 20, 22 becomes 21 etc.
        sort=sort=np.argsort(nodearr)[0:4]
        neighbors=np.where(sort>i-1,sort+1,sort)
        for neighbor in neighbors:
            edge=[i,neighbor]
            edge_index.append(edge)
    return edge_index
def make_graph(Event_number,connection,feat_list,target_list,SRT,N_neighbors):
    """Pass event number, the connection to the database, a string of features and a string of targets, along with 1 for SRTcleaned and 0 for not"""
    if SRT==1:
        SRT="and SRTInIcePulses = 1"
    else:
        SRT=""
    query_features = 'select %s from features WHERE event_no = %s %s '%(feat_list,str(Event_number),SRT)
    features = pd.read_sql(query_features, connection)
    query_truth = 'select %s from truth WHERE event_no = %s '%(target_list,str(Event_number))
    truth = pd.read_sql(query_truth, connection)
    
    "do operations on retrieved dataframes and create tensors, check if datatypes are creating errors"
    features=features.to_numpy()
    edge_index=NearestNeighborEdges(features, N_neighbors)


    features=torch.tensor(features)
    truth=torch.tensor(truth.values)
    edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
    return Data(x=features,edge_index=edge_index,y=truth)
def Graphlist(Event_list,connection,feat_list,target_list,SRT,N_neighbors):
    "Uses eventlist to return a list of graph Dataobjects"
    graphlist=list()
    for i in range(len(Event_list)):
        graph=make_graph(Event_list[i],connection,feat_list,target_list,SRT,N_neighbors)
        graphlist.append(graph)
    return graphlist
def Savejob(Eventlist,datafilename,feat_list,target_list,SRT,N_neighbors,Jobid,filefolder):
    "function to save files to filefolder. Splits them so as to not have too much in memory"
    connection=sqlite3.connect(datafilename)
    graph_per_file=4000
    Ntot=len(Eventlist)
    currpos=0
    splits=np.floor(Ntot/graph_per_file).astype("int32")
    for i in range(splits):
        currpos+=1
        start=i*graph_per_file
        end=graph_per_file*(1+i)
        Events_to_save=Eventlist[start:end]
        print(i,len(Events_to_save))
        graphs=Graphlist(Events_to_save, connection, feat_list, target_list, SRT, N_neighbors)
        filename="graph_job%s_file%s.pkl" % (Jobid,i)
        file=open(filefolder+filename,'wb')
        torch.save(graphs,file)
    "this part takes care of the remainders after splitting into equal sizes"
    if splits<Ntot/graph_per_file:
        start=splits*graph_per_file
        Events_to_save=Eventlist[start:]
        graphs=Graphlist(Events_to_save, connection, feat_list, target_list, SRT, N_neighbors)
        filename="graph_job%s_file%s.pkl" % (Jobid,splits+1)
        file=open(filefolder+filename,'wb')
    torch.save(graphs,file)
if __name__ == '__main__':
    
    datafolder="D:\Schoolwork\Masters\Data\\"
    currentset="dev_upgrade_train_step_001"
    transformers=datafolder+currentset+"/"+"NewSKlearntransformers.pkl"
    transforms=open(transformers,'rb')
    transformdict=pickle.load(transforms)
    datafilename=datafolder+currentset+"/"+"Data.db"
    graph_folder="D:\Schoolwork\Masters\Data\\Graphs\\"+currentset+"/"
    con=sqlite3.connect(datafilename)
    feat_list="dom_x,dom_y,dom_z,time"
    target_list="energy_log10,time,position_x,position_y,position_z,direction_x,direction_y,direction_z,azimuth,zenith"
    
    N_events=12000
    N_events_max=5731206
    N_neighbors=4
    # query_events="select DISTINCT event_no from features"
    # totevents=pd.read_sql(query_events,con).squeeze()
    
    query_events="select DISTINCT event_no from features LIMIT %d" % (N_events)
    Events =pd.read_sql(query_events,con).squeeze()
    SRT=1
    Events=Events.to_numpy()
    # Savejob(Events, datafilename, feat_list, target_list, SRT, N_neighbors, 1, graph_folder)

    N_jobs=4
    N_per_job=np.floor(N_events/N_jobs).astype("int32")
    Processes = []
    for i in range(N_jobs):
        start=i*N_per_job
        end=N_per_job*(1+i)
        Job_events=Events[start:end]
        Processes.append(torch.multiprocessing.Process(target=Savejob, args=([Job_events,datafilename,feat_list,target_list,SRT,N_neighbors,i,graph_folder])))
    start=time.time()
    for process in Processes:
        process.start()
    for process in Processes:
        process.join()
    print (start-time.time())