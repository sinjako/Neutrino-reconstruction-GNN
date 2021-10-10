# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:16:22 2021

@author: Haider
"""
import time
from datetime import datetime
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing
import Processing
import pickle
from models import dynedge, GRUconv,simpedge,dynedgecontext,dynedgeglobvar,dynedgeGlobatt,dynedgeEdgepool,dynedgeSAG,dynedgeTopk,dynedgegauss,dynedgegauss_edgepool
from loss_functions import logcosh,VonMisesSineCosineLoss,gaussloss
from torch_geometric.data import DataLoader
import pandas as pd
def Trainloop(model,optimizer,lossfunction,targetvar,device,train_graphs,val_graphs,workers,batch_size,train_batches,val_batches,epochs):
    average_train_loss_per_epoch = list()
    average_val_loss_per_epoch=list()
    for epoch in tqdm(range(epochs)):

        deadcheck=False
        valcheck=False
        model.train()
        train_loss = torch.tensor([0],dtype = float).to(device)
        if epoch==0:
                Manager=torch.multiprocessing.Manager()
                train_queue=Manager.Queue()
                val_queue=Manager.Queue()
                train_queue_fill=Processing.Spawn_Processes(workers, train_graphs, train_queue, batch_size)
        for k in range(train_batches):
            with torch.enable_grad():
                model.train()
                batch=Processing.GrabBatch(train_queue,device)
                optimizer.zero_grad()
                output=model(batch)
                # print (output)
                # print (batch.x,output.size(),batch)
                # print(output.size())
                loss=lossfunction(output,batch.y[:,targetvar])
                # print (loss)
                loss.backward()
                optimizer.step()
                # print (loss,output,batch.y[:,targetvar])

            deadcheck=Processing.Process_check(train_queue_fill,deadcheck,k,train_batches)
            train_loss +=loss

            if deadcheck==True and valcheck==False:
                val_queue_fill=Processing.Spawn_Processes(workers,val_graphs,val_queue,batch_size)
                valcheck=True
            # print (k)
            if(torch.sum(torch.isnan(output)) != 0):
                raise TypeError('NAN ENCOUNTERED AT : %s / %s'%(k,train_batches))
        if( deadcheck == False):
                print('Training Processes Still Alive. Terminating...')
                for process in train_queue_fill:
                    process.terminate()
                val_queue_fill=Processing.Spawn_Processes(workers,val_graphs,val_queue,batch_size)
        if deadcheck==True and valcheck==False:

                valcheck=True
        with torch.no_grad():
            val_loss,train_queue_fill=Validation(model,optimizer,lossfunction,targetvar,train_graphs,val_graphs,train_queue,val_queue,val_queue_fill,train_queue_fill,workers,batch_size,val_batches,epoch,epochs)
        average_train_loss_per_epoch.append(train_loss.item()/(train_batches*batch_size))
        average_val_loss_per_epoch.append(val_loss.item()/(val_batches*batch_size))
        print (train_loss.item()/(train_batches*batch_size))
    deadcheck=Processing.Process_check(train_queue_fill,deadcheck,k,train_batches) 
    if( deadcheck == False):
        print('Training done. Terminating slaves...')
        for process in train_queue_fill:
            process.terminate()
    del batch,loss,train_loss,val_loss
    return model,average_train_loss_per_epoch,average_val_loss_per_epoch
def Validation(model,optimizer,lossfunction,targetvar,train_graphs,val_graphs,train_queue,val_queue,val_queue_fill,train_queue_fill,workers,batch_size,val_batches,epoch,epochs):
    deadcheck=False
    traincheck=False
    val_loss=torch.tensor([0],dtype = float).to(device)
    for k in range(val_batches):
        with torch.no_grad():
            batch=Processing.GrabBatch(val_queue,device)
            output=model(batch)
            loss=lossfunction(output,batch.y[:,targetvar])
        val_loss+=loss
        deadcheck=Processing.Process_check(val_queue_fill,deadcheck,k,val_batches)
        if deadcheck==True and traincheck==False and epoch<epochs-1 :
            train_queue_fill=Processing.Spawn_Processes(workers, train_graphs, train_queue, batch_size)
            traincheck=True
    deadcheck=Processing.Process_check(val_queue_fill,deadcheck,k,val_batches)
    if deadcheck == False:
            print('Validation Processes Still Alive. Terminating...')
            for process in val_queue_fill:
                process.terminate()
            if epoch<epochs -1:
                train_queue_fill=Processing.Spawn_Processes(workers, train_graphs, train_queue, batch_size)

    return val_loss,train_queue_fill
def Predict(model,prediction_graphs,workers,pred_mini_batches,batch_size,currfolder,device):
    # print('PREDICTING: \n \
    #       model   : %s \n \
    #       n_events: %s' %(baseline,pred_mini_batches*batch_size))
    predictions     = []
    variances       = []
    truths          = []
    pred_events     = []
    manager         = torch.multiprocessing.Manager()
    q               = manager.Queue()
    slaves          = Processing.Spawn_Processes(workers, prediction_graphs, q,batch_size)
    dead_check      = False
    model.eval()
    with torch.no_grad():
        for mini_batch in range(0,pred_mini_batches):
            data            = Processing.GrabBatch(q,device)
            output          = model(data)
            prediction      = output[:,0]
            variance        = output[:,1]
            truth           = data.y[:,0].unsqueeze(1).detach().cpu().numpy()
            pred_events.extend(data.event_no.detach().cpu().numpy())
            predictions.extend(prediction.detach().cpu().numpy())
            variances.extend(variance.detach().cpu().numpy())
            truths.extend(truth)
            dead_check =Processing.Process_check(slaves, dead_check, mini_batch, pred_mini_batches)

        if( dead_check == False):
            for slave in slaves:
                slave.terminate()
        print('Saving results...')
        truths          = pd.DataFrame(truths)
        predictions     = pd.DataFrame(predictions)
        predictionsvar  = pd.DataFrame(variances)
        pred_events     = pd.DataFrame(pred_events)
        result          = pd.concat([pred_events,truths, predictions,predictionsvar],axis = 1)
        result.columns  = ['event_no','E','E_pred','E_pred_var']
        result.to_csv(currfolder + 'predictions.csv',index=False)

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seqlength=3
    input_size=4
    # model = GRUconv(seqlength=seqlength,input_size=input_size).to(device)
    # modelinfo="GRUCONV(seqlength=%d,input_size=%d)" % (seqlength,input_size)
    model =dynedgegauss_edgepool(mode='gauss').to(device)
    modelinfo="dynedgegauss_edgepool(mode='gauss')"
    # model=simpedge().to(device)
    # modelinfo="simpedge"
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps= 1e-3)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    currentset="dev_numu_train_l5_retro_001\\"
    graph_folder="0_nearest_xyzt_alltargets\\"
    targetfolder="D:\Schoolwork\Masters\\Data\\"+currentset+"Graphs\\"+graph_folder
    resultfolder=targetfolder+"Results\\"
    # transformers=
    epochs=10
    defaultdescription=" work on optimizing dynedge. Predicting on Energy.Working on dev_numu_train_l5_retro_001, this has probabilistic regression"
    foldername=input("Type in a folder name. If left blank, it will default to model info which is :"+modelinfo+": Also, you are working on the following graph : "+graph_folder)
    if foldername=="":
        foldername=modelinfo
    description=input("Describe this run, press enter immediately for default description which is : " + defaultdescription+": ")
    if description=="":
        description=defaultdescription
    workers=4
    batch_size=500
    graph_per_file=4000
    val_graphs=list()
    train_graphs=list()
    Nfiles=32 # chosen so the last partial file is excluded, choose 32 and 26 for 500k nodes
    valfilesind=26 # where you switch to loading into validation graph
    train_batches=int(valfilesind*graph_per_file*workers/batch_size)
    val_batches=int((Nfiles-valfilesind)*graph_per_file*workers/batch_size)
    targetvar=0
    lossfunction=gaussloss
    for i in range(workers):

        val_graphs_temp=list()
        train_graphs_temp=list()
        for j in range(Nfiles):
            string=targetfolder + "graph_job%s_file%s.pkl" % (i,j)
            if j>=valfilesind:

                val_graphs_temp.append(string)

            else:

                train_graphs_temp.append(string)

        val_graphs.append(val_graphs_temp)
        train_graphs.append(train_graphs_temp)

    
    model,train_loss,val_loss=Trainloop(model,optimizer,lossfunction,targetvar,device,train_graphs,val_graphs,workers,batch_size,train_batches,val_batches,epochs)
  


    
    
    

    "RANDOM SHIT HERE BOYS!!!!"

   
    trainstring="train_loss_%sbatchsize_%strain_%s_val_%sepochs_%s.pt" %(batch_size,train_batches,val_batches,epochs,modelinfo)
    valstring="val_loss_%sbatchsize_%strain_%s_val_%sepochs_%s.pt" %(batch_size,train_batches,val_batches,epochs,modelinfo)
    modelstring="model_%sbatchsize_%strain_%s_val_%sepochs_%s.pt" %(batch_size,train_batches,val_batches,epochs,modelinfo)
 
    savefolder="D:\Schoolwork\Masters\Script\Savefolder\\"
    now=datetime.now()
    now_string=now.strftime("%d-%m-%Y-%H-%M-%S")
    dirstring=savefolder+now_string+"-"+foldername+"\\"
    os.makedirs(dirstring)
    
    plotstring=""
    torch.save(train_loss,open(dirstring+trainstring,'wb')) # this saves into the datetime folder, to keep track of iterations for later
    torch.save(val_loss,open(dirstring+valstring,'wb'))
    torch.save(description,open(dirstring+"description.txt",'wb'))
    "loss function plots"
    fig,axs=plt.subplots(2,1,constrained_layout=True)
    axs[0].plot(train_loss,'o',linestyle='dotted')
    axs[0].set_title('Average log cosh of error in training')
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Log cosh(predicted-True)")
    fig.suptitle(modelinfo + ' performance in training')
    axs[1].plot(val_loss,'o',linestyle='dotted')
    axs[1].set_title('Average log cosh of error in validation')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Log cosh(predicted-True)")
    figstring="figure_%sbatchsize_%strain_%s_val_%sepochs_%s.png" %(batch_size,train_batches,val_batches,epochs,modelinfo)
    fig.savefig(dirstring+figstring)
    
    "Prediction and save model, be wary of uncommenting since it will make the folder huge"
    Evalfiles=32
    eval_graphs=list()
    evalbatches=int(Evalfiles*graph_per_file*workers/batch_size)
    for i in range(workers):
        eval_graphs_temp=list()
        for j in range(Evalfiles):
            string=targetfolder + "graph_job%s_file%s.pkl" % (i,j+Nfiles)#Nfiles so we predict on other events than training and validation set
            eval_graphs_temp.append(string)


        eval_graphs.append(eval_graphs_temp)
    Predict(model,eval_graphs,workers,evalbatches,batch_size,dirstring,device) # not commenting this out takes both a fuckton of space AND TIME
    state={
    'modelinfo': modelinfo,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    }
    torch.save(state,open(dirstring+modelstring,'wb')) ## ONLY UNCOMMENT IF YOU NEED TO SAVE MODEL, OTHERWISE THIS WILL TAKE ALOT OF SPACE