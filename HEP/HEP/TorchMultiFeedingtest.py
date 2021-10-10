# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:16:22 2021

@author: Haider
"""
import time
import torch
import numpy as np
import matplotlib as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import CrossEntropyLoss,Linear
from torch_geometric.nn.pool import avg_pool
from torch_geometric.nn.glob import global_mean_pool
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

import torch.multiprocessing

import Processing

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16,2)
        self.linear=Linear(2, 2, bias=True)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x =self.linear(x)
        x=global_mean_pool(x,data.batch)
        return x
if __name__ == '__main__':
    
    targetfolder="D:\Schoolwork\Masters\Data\Graphs\\testonly\\"
    # loadobject=torch.load(open(targetfolder +"graphs_job0_file0.pkl","rb"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    epochs=2
    workers=4
    batch_size=32
    val_graphs=list()
    train_graphs=list()
    minibatches=2528
    valbatches=312
    for i in range(4):
        string=targetfolder+"graphs_job%s_file9.pkl" % (i)
        val_graphs_temp=list()
        val_graphs_temp.append(string)
        val_graphs.append(val_graphs_temp)
        train_graphs_job=list()
        for j in range(9):
            string=targetfolder + "graphs_job%s_file%s.pkl" % (i,j)
            train_graphs_job.append(string)
        train_graphs.append(train_graphs_job)

    Manager=torch.multiprocessing.Manager()
    train_queue=Manager.Queue()
    val_queue=Manager.Queue()
    lossfunction=CrossEntropyLoss().to(device)
    lossarr=np.zeros(epochs)
    for epoch in tqdm(range(epochs)):
        train_queue_fill=Processing.Spawn_Processes(workers, train_graphs, train_queue, batch_size)
        deadcheck=False
        valcheck=False
        model.train()
        losstemp=np.zeros(valbatches)
        for k in range(minibatches):
        
            batch=Processing.GrabBatch(train_queue,device)
            optimizer.zero_grad()
            output=model(batch)
            loss=lossfunction(output,batch.y)
            loss.backward()
            optimizer.step()
                
            deadcheck=Processing.Process_check(train_queue_fill,deadcheck,k,minibatches)
            if deadcheck==True and valcheck==False:
                print (deadcheck)
                val_queue_fill=Processing.Spawn_Processes(workers,val_graphs,val_queue,batch_size)
                valcheck=True
        model.eval()
        for k in range(valbatches):
            batch=Processing.GrabBatch(val_queue,device)
            output=model(batch)
            loss=lossfunction(output,batch.y)
            losstemp[k]=loss.item()
        lossarr[epoch]=np.mean(losstemp)
        print ("done training")
        # while count<workers:
        #     batch=GrabBatch(val_queue)
        #     if batch=="done":
        #         count+=1
        #     else:
        
        
    