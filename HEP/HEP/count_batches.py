import time
from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing
import Processing
import pickle
from models import dynedge, GRUconv,simpedge,dynedgecontext,dynedgeglobvar,dynedgeGlobatt,dynedgeEdgepool,dynedgeSAG,dynedgeTopk
from loss_functions import logcosh,VonMisesSineCosineLoss
from torch_geometric.data import DataLoader
import pandas as pd
import copy
from torch.serialization import load

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    device = 'cpu'
    seqlength=3
    input_size=4
    graph_job=4
    currentset="dev_upgrade_train_step_001/"
    graph_folder="4_nearest_xyzt_alltargets/"
    targetfolder="/groups/hep/haider/Masters/Data/"+currentset+"Graphs/"+graph_folder
    Nfiles=358 # chosen so the last partial file is excluded, choose 32 and 26 for 500k nodes
    valfilesind=328 # where you switch to loading into validation graph
    #330 and 300 for full, 358 and 328for upgrade
    workers=2
    val_graphs=list()
    train_graphs=list()
    batch_size=500
    graph_per_file=4000
    train_batches=int(valfilesind*graph_per_file*graph_job/batch_size)
    val_batches=int((Nfiles-valfilesind)*graph_per_file*graph_job/batch_size)
    for i in range(graph_job):

        val_graphs_temp=list()
        train_graphs_temp=list()
        for j in range(Nfiles):
            string=targetfolder + "graph_job%s_file%s.pkl" % (i,j)
            if j>=valfilesind:

                val_graphs.append(string)

            else:

                train_graphs.append(string)
    val_graphs=np.array_split(val_graphs,1)
    train_graphs=np.array_split(train_graphs,1)
    count=0
    print (val_graphs)
    for i in val_graphs:
        for j in i:
            data_list_objects = load(j)
            loader = DataLoader(data_list_objects,batch_size = batch_size,drop_last=True,pin_memory=True)
            count+=len(loader)
    print (count)
    print (train_graphs)
    for i in train_graphs:
        for j in i:
            data_list_objects = load(j)
            loader = DataLoader(data_list_objects,batch_size = batch_size,drop_last=True,pin_memory=True)
            count+=len(loader)
    print (count)