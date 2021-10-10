# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:46:13 2021

@author: Haider
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sqlite3
from datetime import datetime
import os
from itertools import product,combinations
def MakePerformancePlots(results,model_name):
   # Produces simple performance plots for regression results on energy_log10
   #
   # ARGUMENTS:
   # results - Pandas DataFrame. 
   # Must contain columns: 'event_no', 'energy_log10', 'energy_log10_pred'
   #    'event_no'         - The event number of the events
   #    'energy_log10'     - The true energy for the events
   #    'energy_log10_pred'- Your regression of the true energy
   #
   # model_name - String. The name of your model to put in the legend of the plots.
   #
   #
   # RETURNS:
   # List() of matplotlib.pyplot.figure() objects
    plotlength=len(results)
    print('Making Performance Plots..')
########################################    
#    
#           FIRST PLOT
#    
########################################
    #### CALCULATIONS FOR FIRST PLOT
    totevents=len(results[0])
    for result in results:
        if len(result) != totevents:
            print ("WARNING, AMOUNT OF EVENTS IN EACH PREDICTION SET IS NOT UNIFORM")
            print (len(results))
    figs = list()
    num_bins = 10
    errorlist=list()
    meanslist=list()
    means_E_list=list()
    for result in results:
        
        result = result.sort_values('event_no').reset_index(drop=True)
        result = result.sort_values('energy_log10')
        E = result['energy_log10']
        E_pred = result['energy_log10_pred']

    
        n, bins, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
        plt.close()
        error = list()
        means = list()
        means_E = list()
        for k in range(len(bins)-1):
            index = (E >= bins[k]) & (E <= bins[k+1])
            means.append(np.mean(E_pred[index]))
            means_E.append(np.mean(E[index]))
            error.append(np.mean(abs(E[index] - E_pred[index])))
        errorlist.append(error)
        meanslist.append(means)
        means_E_list.append(means_E)
        
        # print("errors",errorlist,"means",meanslist,"means_E_list",means_E_list)
    ### MAKING FIRST PLOT    
    fig2=plt.figure()
    host2 = fig2.add_subplot(111)
    par2 = host2.twinx()
    host2.set_xlabel('$Energy_{True}$',size = 20)
    host2.set_ylabel('Energy$_{pred}$', size = 20)
    par2.set_ylabel('Count',size = 20)
    host2.plot(means_E,means_E)
    for i in range(plotlength):
        host2.errorbar(means_E_list[i],meanslist[i],errorlist[i],linestyle='dotted',fmt = 'o',markersize=3,capsize = 8,label=model_name[i])
    num_bins = 20
    n, bins, patches = par2.hist(E, num_bins, facecolor='blue', alpha=0.1,label = model_name)   # no need for this in loop since this is histogram of events 

    host2.legend()
    plt.title('True vs Predicted', size = 20)
    figs.append(fig2)
    plt.close()
    
    
    
###############################################

# SECOND AND THIRD PLOT

###############################################



    num_bins = 10
    fig3 = plt.figure()
    n, bins, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
    plt.close()
    errors_list=list()
    errors_width_list = list()
    width_list       = list()
    means_log_list = list()
    medians_E_list = list()
    for result in results:
        errors_width = list()
        width       = list()
        means_log = list()
        medians_E = list()
        means_E = list()
        result = result.sort_values('event_no').reset_index(drop=True)
        result = result.sort_values('energy_log10')
        E = result['energy_log10']
        E_pred = result['energy_log10_pred']
        for k in range(len(bins)-1):
            index = (E >= bins[k]) & (E<= bins[k+1])
            if(sum(index) != 0):
                means_log.append(np.mean(E_pred[index]/E[index]))
                means_E.append(np.mean(E[index]))
                
                medians_E.append(np.median(E_pred[index]/E[index]))
                diff = (E_pred / E)[index].reset_index(drop = True)
                
                N = sum(index)
            
                x_25 = abs(diff-np.percentile(diff,25,interpolation='nearest')).argmin() #int(0.16*N)
                x_75 = abs(diff-np.percentile(diff,75,interpolation='nearest')).argmin() #int(0.84*N)
            
                fe_25 = sum(diff <= diff[x_25])/N # This is for third plot
                fe_75 = sum(diff <= diff[x_75])/N # This is for third plot
                errors_width.append(np.sqrt((0.25*(1-0.25)/N)*(1/fe_25**2 + 1/fe_75**2))*(1/1.349)) # This is for third plot
            
                if( k == 0):
                    errors = np.array([np.median(diff) - diff[x_25],            
                                        np.median(diff) - diff[x_75]])      # This is for second plot
                    width = np.array(-diff[x_25]+ diff[x_75])/1.349        # This is for third plot
                    
                else:
                        errors = np.c_[errors,np.array([np.median(diff) - diff[x_25],
                                                        np.median(diff) - diff[x_75]])] # This is for second plot 
                        width = np.r_[width,np.array(-diff[x_25]+ diff[x_75])/1.349]    # This is for third plot
        errors_width_list.append(errors_width)
        width_list.append(width)
        means_log_list.append(means_log) 
        medians_E_list.append(medians_E)
        errors_list.append(errors)
    #### MAKING SECOND PLOT
    ylabel = r'median($E_{pred}$-E)'
    title = r' Median vs E '
    fig=plt.figure()
    host = fig.add_subplot(111)
    par1 = host.twinx()
    host.set_xlabel('log(E) [GeV]',size = 20)
    host.set_ylabel(ylabel, size = 20)
    par1.set_ylabel('Count',size = 20)
    n, bins, patches = par1.hist(E, num_bins, facecolor='blue', alpha=0.1,label = '_nolegend_')        
    #host.scatter(means_E,means_log_list[j],color = 'red', s = 30,label = '_nolegend_')
    for i in range(plotlength):
        host.errorbar(means_E,medians_E_list[i],abs(errors_list[i]),linestyle='dotted',fmt = 'o',markersize=3,capsize = 8, label = model_name[i])
    host.legend()
    #host.set_ylim([-1.30,0.6])
    host.grid()
    plt.title(title, size = 20)
    figs.append(fig)
    
    plt.close()
    
   

    ##### MAKING THIRD PLOT
    fig7=plt.figure()
    host = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=3)
    par1 = host.twinx()
    host.set_ylabel( r'W((${E_{pred}}-{E}$))')#, size = 20)
    host.set_xlabel('Log(E) [GeV])')#,size = 20)
    #host.tick_params(axis='both', which='major', labelsize=20)
    #host.tick_params(axis='both', which='minor', labelsize=20)
    par1.set_ylabel('Events',size = 20)
    n, bins, patches = par1.hist(E, num_bins, facecolor='blue', alpha=0.1,label = None)
    for i in range(plotlength):
        host.errorbar(means_E,list(width_list[i]),errors_width_list[i],linestyle='dotted',fmt = 'o',markersize=3,capsize = 8, label = model_name[i])
    #host.yaxis.set_ticks(np.arange(0,1.1,0.1))
    host.legend()
    plt.title( r'W((${E_{pred}}-{E}$)) vs E', size = 20)
    host.grid()
    figs.append(fig7)
    plt.close() 


    ###### MAKING FOURTH PLOT 
    fig8=plt.figure()
    host = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=3)
    host.set_ylabel( r'Count')#, size = 20)
    host.set_xlabel('Log(E) [GeV])')#,size = 20)
    num_bins=50
    E_list=list()
    E=results[0]['energy_log10']
    E_list.append(E.to_numpy())
    labels=list()
    labels.append("True")
    labels.extend(model_name)
    for i in range(plotlength):
        result = results[i].sort_values('event_no').reset_index(drop=True)
        result = result.sort_values('energy_log10')
        E_pred = result['energy_log10_pred'].to_numpy()
        E_list.append(E_pred)

    host.hist(E_list,num_bins,label=labels,histtype='step')

    host.legend()
    plt.title( r'Energy Histogram', size = 20)
    host.grid()
    figs.append(fig8)
    plt.close()
    ######## FIFTH PLOT
    
    fig9=plt.figure()
    nbins=50
    host = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=3)
    host.set_ylabel( r'Count')#, size = 20)
    host.set_xlabel('Log(E) [GeV])')#,size = 20)
    residlist=list()
    labels=list()
    for i in range(plotlength):
        labels.append(model_name[i])
        result = results[i].sort_values('event_no').reset_index(drop=True)
        result = result.sort_values('energy_log10')
        E_pred = result['energy_log10_pred'].to_numpy()
        E=result['energy_log10'].to_numpy()
        resid=abs(E_pred-E)
        residlist.append(resid)
    percentile=99
    rmax=np.percentile(residlist[0],percentile)
    host.hist(residlist,num_bins,label=labels,histtype='step',range=(0,rmax))
    plt.title(r'Residual histogram,percentile=%s' %(percentile),size=20)
    host.legend()
    host.grid()
    figs.append(fig9)
    plt.close()
    return figs    

def grabdata(path,transformer):
    path_to_file=path+"\\predictions.csv"
    data=pd.read_csv(path_to_file).reset_index(drop=True)
    data=data[["event_no",'E','E_pred']]
    data.columns=["event_no","energy_log10","energy_log10_pred"]
    transform= "energy_log10"
    data["energy_log10"]=transformer["truth"][transform].inverse_transform(data[["energy_log10"]])
    data["energy_log10_pred"]=transformer["truth"][transform].inverse_transform(data[["energy_log10_pred"]])
    return data
    
transformfile=open("D:\Schoolwork\Masters\Data\dev_numu_train_l5_retro_001\\transformers.pkl",'rb')
transformer=pickle.load(transformfile)
paths=list()
results=list()
model_name=list()

paths.append("D:\Schoolwork\Masters\Script\Savefolder\\18-05-2021-09-49-04-dynedge(k=[4,4,4,4])")
model_name.append("Dynedge")

paths.append("D:\Schoolwork\Masters\Script\Savefolder\\01-06-2021-01-25-24-dynedgeGlobatt")
model_name.append("DynedgeglobAtt")

paths.append("D:\Schoolwork\Masters\Script\Savefolder\\01-06-2021-01-41-15-dynedgeGlobatt_NoNN4")
model_name.append("dynedgeGlobatt_NoNN4")

paths.append("D:\Schoolwork\Masters\Script\Savefolder\\01-06-2021-02-36-51-dynedgeglobatt_nonn4_relu")
model_name.append("dynedgeglobatt_nonn4_relu")

paths.append("D:\Schoolwork\Masters\Script\Savefolder\\01-06-2021-03-12-26-dynedgepool1")
model_name.append("01-06-2021-03-12-26-dynedgepool1")

paths.append("D:\Schoolwork\Masters\Script\Savefolder\\01-06-2021-04-00-25-dynedgeSAG")
model_name.append("01-06-2021-04-00-25-dynedgeSAG")

paths.append("D:\Schoolwork\Masters\Script\Savefolder\\01-06-2021-04-27-32-dynedgeTopk")
model_name.append("01-06-2021-04-27-32-dynedgeTopk")

# paths.append("D:\Schoolwork\Masters\Script\Savefolder\\18-05-2021-10-23-53-dynedge_context1")
# model_name.append("context_add")

# paths.append("D:\Schoolwork\Masters\Script\Savefolder\\18-05-2021-11-10-53-dynedge_context1_big")
# model_name.append("context_add_big")

# paths.append("D:\Schoolwork\Masters\Script\Savefolder\\18-05-2021-10-52-04-dynedge_context2")
# model_name.append("context_concat")

# paths.append("D:\Schoolwork\Masters\Script\Savefolder\\25-05-2021-00-44-56-dynedgecontextconcatrelu")
# model_name.append("context_concat_relu")

# paths.append("D:\Schoolwork\Masters\Script\Savefolder\\25-05-2021-23-51-04-dyncontextnoact")
# model_name.append("context_concat_noact")

# paths.append("D:\Schoolwork\Masters\Script\Savefolder\\26-05-2021-00-23-35-dynedgecontexttanh")
# model_name.append("context_concat_tanh")

for path in paths:
    results.append(grabdata(path,transformer))
figs=list()
newmodels=list()
newpaths=list()
newresults=list()

percombo=2
for combo in combinations(model_name,percombo):
    newmodels.append(combo)
for combo in combinations(paths,percombo):
    newpaths.append(combo)
for combo in combinations(results,percombo):
    newresults.append(combo)
my_cool_figures=list()
for i in range(len(newmodels)):
    compfigs=MakePerformancePlots(newresults[i],newmodels[i])
    my_cool_figures.extend(compfigs)

figname=input("Do you want to save this plot? If yes, type in the name given to the folder. If no, leave blank and just press enter ")
if figname != "":
    description=input("You have decided to save the plot using figure name :"+ figname +" : now type in a description for the plots ")
    now=datetime.now()
    now_string=now.strftime("%d-%m-%Y-%H-%M-%S")
    savefolder="D:\Schoolwork\Masters\Script\Savefolder\\"
    dirstring=savefolder+now_string+"-"+figname+"\\"
    os.makedirs(dirstring)
    descfile=open(dirstring+"description.txt",'wb')
    pickle.dump(description,descfile)
    infos=list()
    infos.append(paths)
    infos.append(model_name)
    infofile=open(dirstring+"info.pkl","wb")
    pickle.dump(infos,infofile)
    count=0
    pair=0
    totplots=5
    for fig in my_cool_figures:
        
        if count==totplots:
            count=0
            pair+=1
        count+=1
        print (pair,count)
        if count==1:
            currstring="Error_%s_%s"%(newmodels[pair][0],newmodels[pair][1])
            fig.savefig(dirstring+currstring)
        if count==2:
            currstring="Error_median_%s_%s"%(newmodels[pair][0],newmodels[pair][1])
            fig.savefig(dirstring+currstring)
        if count==3:
            currstring="Error_width__%s_%s"%(newmodels[pair][0],newmodels[pair][1])
            fig.savefig(dirstring+currstring)
        if count==4:
            currstring="Energy_distr__%s_%s"%(newmodels[pair][0],newmodels[pair][1])
            fig.savefig(dirstring+currstring)
        if count==5:
            currstring="resid_distr__%s_%s"%(newmodels[pair][0],newmodels[pair][1])
            fig.savefig(dirstring+currstring)

# for fig in my_cool_figures:
#     count+=1
#     fig.savefig(path_to_folder+"\\predplot%d.png"%(count)) 
"retro plots down here"
# path_to_retro="D:\Schoolwork\Masters\Data\dev_numu_train_l5_retro_001\\predictions.db"
# con=sqlite3.connect(path_to_retro)
# retro = pd.read_sql_query('SELECT event_no,energy_log10 from retro_reco', con) 
# retro.columns=["event_no","energy_log10_pred"]
# path_to_database="D:\Schoolwork\Masters\Data\dev_numu_train_l5_retro_001\\Data.db"
# con2=sqlite3.connect(path_to_database)
# truevals=pd.read_sql_query('SELECT event_no,energy_log10 from truth where interaction_type=1', con2)
# truevals["energy_log10"]=transformer["truth"]["energy_log10"].inverse_transform(truevals[["energy_log10"]])

# combined=pd.merge(truevals,retro,on="event_no")

# #do the infinity checks after combining to minimize time spent on it

# infcheck=np.isinf(combined['energy_log10_pred'])
# combined=combined[~infcheck]
# combined["energy_log10_pred"]=transformer["truth"]["energy_log10"].inverse_transform(combined[["energy_log10_pred"]])
# combined["energy_log10"]=transformer["truth"]["energy_log10"].inverse_transform(combined[["energy_log10"]]) # 
# lame_figures=MakePerformancePlots(combined,'retro')