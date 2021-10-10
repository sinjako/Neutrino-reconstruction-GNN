# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:16:22 2021

@author: Haider
"""
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
def Trainloop(model,optimizer,lossfunction,targetvar,device,train_graphs,val_graphs,workers,batch_size,train_batches,val_batches,epochs):
    average_train_loss_per_epoch = list()
    average_val_loss_per_epoch=list()
    models=list()
    besterr=10000000
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
            #print (k,train_queue.qsize())
            with torch.enable_grad():
                model.train()
                batch=Processing.GrabBatch(train_queue,device)
                optimizer.zero_grad()
                output=model(batch)[:,0]
                #print ("donezo",k)
                loss=lossfunction(output,batch.y[:,targetvar])
                loss.backward()
                optimizer.step()
                # print (loss,output,batch.y[:,targetvar])

            deadcheck=Processing.Process_check(train_queue_fill,deadcheck,k,train_batches)
            train_loss +=loss
            if deadcheck==True and valcheck==False:
                val_queue_fill=Processing.Spawn_Processes(workers,val_graphs,val_queue,batch_size)
                valcheck=True
        
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
        

        if average_val_loss_per_epoch[-1]<besterr:
            bestmodel=copy.deepcopy(model)
            besterr=average_val_loss_per_epoch[-1]
    deadcheck=Processing.Process_check(train_queue_fill,deadcheck,k,train_batches) 
    if( deadcheck == False):
        print('Training done. Terminating slaves...')
        for process in train_queue_fill:
            process.terminate()
    del batch,loss,train_loss,val_loss,model
    return bestmodel,average_train_loss_per_epoch,average_val_loss_per_epoch,besterr
def Validation(model,optimizer,lossfunction,targetvar,train_graphs,val_graphs,train_queue,val_queue,val_queue_fill,train_queue_fill,workers,batch_size,val_batches,epoch,epochs):
    deadcheck=False
    traincheck=False
    val_loss=torch.tensor([0],dtype = float).to(device)
    for k in range(val_batches):
        with torch.no_grad():
            batch=Processing.GrabBatch(val_queue,device)
            output=model(batch)[:,0]
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
def Predict(model,prediction_graphs,workers,pred_mini_batches,batch_size,currfolder,device,k):
    # print('PREDICTING: \n \
    #       model   : %s \n \
    #       n_events: %s' %(baseline,pred_mini_batches*batch_size))
    predictions     = []
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
            prediction      = model(data)
            truth           = data.y[:,0].unsqueeze(1).detach().cpu().numpy()
            pred_events.extend(data.event_no.detach().cpu().numpy())
            predictions.extend(prediction.detach().cpu().numpy())
            truths.extend(truth)
            dead_check =Processing.Process_check(slaves, dead_check, mini_batch, pred_mini_batches)

        if( dead_check == False):
            for slave in slaves:
                slave.terminate()
        print('Saving results...')
        truths          = pd.DataFrame(truths)
        predictions     = pd.DataFrame(predictions)
        pred_events     = pd.DataFrame(pred_events)
        result          = pd.concat([pred_events,truths, predictions],axis = 1)
        result.columns  = ['event_no','E','E_pred']
        result.to_csv(currfolder + 'predictions%s.csv'%(k),index=False)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seqlength=3
    input_size=4
 

    modelinfo="dynedgeedgepool(k=4:12)_1kchange_per_run"
    currentset="dev_numu_train_l5_retro_001/"
    graph_folder="0_nearest_xyzt_alltargets/"
    targetfolder="/groups/hep/haider/Masters/Data/"+currentset+"Graphs/"+graph_folder
    resultfolder=targetfolder+"Results/"
    # transformers=
    epochs=20
    defaultdescription=" work on optimizing dynedge. Predicting on Energy.Working on dev_numu_train_l5_retro_001, working on graph architectures"
    foldername=input("Type in a folder name. If left blank, it will default to model info which is :"+ modelinfo +": Also, you are working on the following graph : "+graph_folder)
    if foldername=="":
        foldername=modelinfo
    description=input("Describe this run, press enter immediately for default description which is : " + defaultdescription+": ")
    if description=="":
        description=defaultdescription
    workers=2
    batch_size=500
    graph_per_file=4000
    val_graphs=list()
    train_graphs=list()
    Nfiles=32 # chosen so the last partial file is excluded, choose 32 and 26 for 500k nodes
    valfilesind=26 # where you switch to loading into validation graph
    graph_job=4
    train_batches=int(valfilesind*graph_per_file*graph_job/batch_size)
    val_batches=int((Nfiles-valfilesind)*graph_per_file*graph_job/batch_size)
    targetvar=0
    lossfunction=logcosh

    for i in range(graph_job):

        for j in range(Nfiles):
            string=targetfolder + "graph_job%s_file%s.pkl" % (i,j)
            if j>=valfilesind:

                val_graphs.append(string)

            else:

                train_graphs.append(string)

    print (len(val_graphs),val_batches,len(train_graphs),train_batches)



    val_graphs=np.array_split(val_graphs,workers)
    train_graphs=np.array_split(train_graphs,workers)
   # print (val_graphs)
   # time.sleep(10000)
    errors=list()
    klist=list()
    currbest=10000000
  #  for i in range(4):
     #   k=[4,4,4,4]
      #  for j in range(4,21):

         #   k[i]=j
         #   print ("now running the following k",k)
          #  currmodel=dynedge(k=k).to(device)
                
         #   optimizer = torch.optim.Adam(currmodel.parameters(), lr=1e-3, eps= 1e-3)
          #  for state in optimizer.state.values():
              #  for k, v in state.items():
                 #   if isinstance(v, torch.Tensor):
                 #       state[k] = v.cuda()
           # currmodel,curr_train_loss,curr_val_loss,besterr=Trainloop(currmodel,optimizer,lossfunction,targetvar,device,train_graphs,val_graphs,workers,batch_size,train_batches,val_batches,epochs)
          #  errors.append(besterr)
         #   klist.append(k)
           # if currbest<besterr:
                #model=copy.deepcopy(currmodel)
               ## besterr=besterr
               # bestk=k
               # train_loss=curr_train_loss
               # val_loss=curr_val_loss
            #del currmodel,curr_train_loss,curr_val_loss,optimizer
    Evalfiles=32
    eval_graphs=list()
    evalbatches=int(Evalfiles*graph_per_file*workers/batch_size)
    for i in range(workers):
        eval_graphs_temp=list()
        for j in range(Evalfiles):
            string=targetfolder + "graph_job%s_file%s.pkl" % (i,j+Nfiles)#Nfiles so we predict on other events than training and validation set
            eval_graphs_temp.append(string)


        eval_graphs.append(eval_graphs_temp)
    savefolder="/groups/hep/haider/Masters/Script/Savefolder/"
    now=datetime.now()
    now_string=now.strftime("%d-%m-%Y-%H-%M-%S")
    dirstring=savefolder+now_string+"-"+foldername+"/"
    os.makedirs(dirstring)
    for i in range(4,12):
        k=[i,i,i,i]
        print ("now running the following k",k)
        currmodel=dynedgeEdgepool(k=k).to(device)
                
        optimizer = torch.optim.Adam(currmodel.parameters(), lr=1e-3, eps= 1e-3)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        currmodel,curr_train_loss,curr_val_loss,besterr=Trainloop(currmodel,optimizer,lossfunction,targetvar,device,train_graphs,val_graphs,workers,batch_size,train_batches,val_batches,epochs)
        Predict(currmodel,eval_graphs,workers,evalbatches,batch_size,dirstring,device,k)
        if currbest<besterr:
            model=copy.deepcopy(currmodel)
            besterr=besterr
            bestk=k
            train_loss=curr_train_loss
            val_loss=curr_val_loss
        errors.append(besterr)
        klist.append(k)
        del currmodel,curr_train_loss,curr_val_loss,optimizer
    


    
    
    

    "RANDOM SHIT HERE BOYS!!!!"

   
    trainstring="train_loss_%sbatchsize_%strain_%s_val_%sepochs_%s.pt" %(batch_size,train_batches,val_batches,epochs,modelinfo)
    valstring="val_loss_%sbatchsize_%strain_%s_val_%sepochs_%s.pt" %(batch_size,train_batches,val_batches,epochs,modelinfo)
    modelstring="model_%sbatchsize_%strain_%s_val_%sepochs_%s.pt" %(batch_size,train_batches,val_batches,epochs,modelinfo)
 

    
    plotstring=""
    torch.save(train_loss,open(dirstring+trainstring,'wb')) # this saves into the datetime folder, to keep track of iterations for later
    torch.save(val_loss,open(dirstring+valstring,'wb'))
    torch.save(description,open(dirstring+"description.txt",'wb'))
    torch.save(bestk,open(dirstring+"bestk.txt","wb"))
    torch.save(errors,open(dirstring+"errors,txt","wb"))
    torch.save(klist,open(dirstring+"klist.txt",wb))
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
# not commenting this out takes both a fuckton of space AND TIME
    state={
    'modelinfo': modelinfo,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    }
    torch.save(state,open(dirstring+modelstring,'wb')) ## ONLY UNCOMMENT IF YOU NEED TO SAVE MODEL, OTHERWISE THIS WILL TAKE ALOT OF SPACE