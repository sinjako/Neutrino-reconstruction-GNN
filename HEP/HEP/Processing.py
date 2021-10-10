# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:34:26 2021

@author: Haider
"""
from torch.serialization import load
import torch.multiprocessing 
from torch_geometric.data import DataLoader
def Process_check(Processes,dead_check,k,mini_batches):
    Process_check = 0
    if dead_check == False:
        for Process in Processes:
            Process_check += Process.is_alive()
        if(Process_check == 0):
            print('All processes have finished at: %s / %s'%(k,mini_batches))
            dead_check = True
            for process in Processes:
                process.terminate()
    else:
        Process_check = 0
    return dead_check
def Spawn_Processes(n_workers, graph_list,q,batch_size):
    Processes = []
    if(n_workers > len(graph_list)):
        n_workers = len(graph_list)
    for j in range(n_workers):
        Processes.append(torch.multiprocessing.Process(target=worker, args=([graph_list[j],q,batch_size])))
    for Process in Processes:
        Process.start()
    print('All task requests sent\n', end='')
    return Processes
def GrabBatch(q,device):
    "grabs batch from queue, puts it on device, hangs if queue is empty"
    # print (device)
    queue_empty = q.empty()
    while(queue_empty):
        # print (queue_empty)
        queue_empty = q.empty()
    mini_batch =q.get()
    if mini_batch=="done":
        return "done"
    else:
        return mini_batch.to(device)
def worker(graph_list,q,batch_size):
    "graphs_train is a list of paths to pkl.files,q is the queue."
    for item in graph_list:
        #print(f'Working on {item}')
        data_list_objects = load(item)
        loader = DataLoader(data_list_objects,batch_size = batch_size,drop_last=True,pin_memory=True)
        loader_it = iter(loader)
        for k in range(0,len(loader)):
            q.put(next(loader_it))
    torch.multiprocessing.current_process().close()
    return