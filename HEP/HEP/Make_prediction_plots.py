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
    
    print('Making Performance Plots..')
########################################    
#    
#           FIRST PLOT
#    
########################################
    #### CALCULATIONS FOR FIRST PLOT
    figs = list()
    results = results.sort_values('event_no').reset_index(drop=True)
    results = results.sort_values('energy_log10')
    E = results['energy_log10']
    E_pred = results['energy_log10_pred']
    num_bins = 20
    
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

        
        
    ### MAKING FIRST PLOT    
    fig2=plt.figure()
    host2 = fig2.add_subplot(111)
    par2 = host2.twinx()
    host2.set_xlabel('$Energy_{True}$',size = 20)
    host2.set_ylabel('Energy$_{pred}$', size = 20)
    par2.set_ylabel('Count',size = 20)
                 
    host2.errorbar(means_E,means,error,linestyle='dotted',fmt = 'o',capsize = 15)
    num_bins = 20
    n, bins, patches = par2.hist(E, num_bins, facecolor='blue', alpha=0.1,label = model_name)   
    host2.plot(means_E,means_E,label = model_name)
    host2.legend()
    plt.title('True vs Predicted', size = 20)
    figs.append(fig2)
    #plt.close()
    
    
    
################################################
#
# SECOND AND THIRD PLOT
#
################################################

    ####### CALCULATIONS FOR SECOND AND THIRD PLOT
    width       = list()
    errors_width = []
    num_bins = 10
    fig3 = plt.figure()
    n, bins, patches = plt.hist(E, num_bins, facecolor='blue', alpha=0.3,label = None)
    plt.close()
    means_log = list()
    means_E = list()
    medians_E = list()
    for k in range(len(bins)-1):
        index = (E >= bins[k]) & (E<= bins[k+1])
        if(sum(index) != 0):
            means_log.append(np.mean(E_pred[index]-E[index]))
            means_E.append(np.mean(E[index]))

            medians_E.append(np.median(E_pred[index]-E[index]))
            diff = (E_pred - E)[index].reset_index(drop = True)
            
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
    host.errorbar(means_E,medians_E,abs(errors),linestyle='dotted',fmt = 'o',capsize = 10, label = model_name)
    host.legend()
    #host.set_ylim([-1.30,0.6])
    host.grid()
    plt.title(title, size = 20)
    figs.append(fig)
    #plt.close()
    
   

    ##### MAKING THIRD PLOT
    fig7=plt.figure()
    host = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=3)
    par1 = host.twinx()
    host.set_ylabel( r'W($({E_{pred}}-{E}$))')#, size = 20)
    host.set_xlabel('Log(E) [GeV])')#,size = 20)
    #host.tick_params(axis='both', which='major', labelsize=20)
    #host.tick_params(axis='both', which='minor', labelsize=20)
    par1.set_ylabel('Events',size = 20)
    n, bins, patches = par1.hist(E, num_bins, facecolor='blue', alpha=0.1,label = None)
           
    host.errorbar(means_E,list(width),errors_width,linestyle='dotted',fmt = 'o',capsize = 10, label = model_name)
    #host.yaxis.set_ticks(np.arange(0,1.1,0.1))
    host.legend()
    plt.title( r'W(({E_{pred}}-{E}$)) vs E', size = 20)
    host.grid()
    figs.append(fig7)
    # plt.close()                  
    return figs    
transformfile=open("D:\Schoolwork\Masters\Data\dev_numu_train_l5_retro_001\\transformers.pkl",'rb')
transformer=pickle.load(transformfile)
path_to_folder="D:\Schoolwork\Masters\Script\Savefolder\\02-06-2021-07-38-43-dynedgegauss(mode='gauss')"
path_to_file=path_to_folder+"\\predictions.csv"
data=pd.read_csv(path_to_file).reset_index(drop=True)
data=data[["event_no",'E','E_pred','E_pred_var']]
data.columns=["event_no","energy_log10","energy_log10_pred",'E_pred_var']
transform= "energy_log10"
data["energy_log10"]=transformer["truth"][transform].inverse_transform(data[["energy_log10"]])
data["energy_log10_pred"]=transformer["truth"][transform].inverse_transform(data[["energy_log10_pred"]])

my_cool_figures = MakePerformancePlots(data,"gauss")
count=0
for fig in my_cool_figures:
    count+=1
    fig.savefig(path_to_folder+"\\predplot%d.png"%(count)) 
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