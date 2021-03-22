# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
from tcn import TCN

import numpy as np
import sqlite3
def tcn_model_26(timesteps, input_dim, output_dim):
    i = tf.keras.Input(batch_shape=(None,timesteps, input_dim))
    o = TCN(
        nb_filters=256,
        kernel_size=2,
        nb_stacks=2,
        dilations=[1, 2, 4, 8, 16,32],
        padding="causal",
        use_skip_connections=True,
        dropout_rate=0.0,
        return_sequences=False,
        activation="relu",
        kernel_initializer="he_normal",
        use_batch_norm=True,
    )(i)
    o = tf.keras.layers.Dense(output_dim)(o)
    model = tf.keras.models.Model(inputs=[i], outputs=[o])
    return model


conn=sqlite3.connect("dev_numu_train_l5_retro_001.db")
# cursor=conn.execute("select name FROM sqlite_master WHERE type='table'")
# print (cursor.fetchall())
curstemp=conn.execute("select * from features")
#print(curstemp.description)
Cursor=conn.execute("SELECT x,y,z,time,charge_log10,event_no from features")
Data=np.zeros(6)
counter=0
for row in Cursor:
    counter=counter+1
    stack=[row[0],row[1],row[2],row[3],row[4],row[5]]
    Data=np.vstack((Data,stack))
    if counter%10000==0:
        print (counter,Data.shape)
        break
Data=np.delete(Data,[0],axis=0)
curstemp2=conn.execute("select * from truth")
#print(curstemp2.description)
Cursor2=conn.execute("SELECT vertex_x,vertex_y,vertex_z,direction_x,direction_y,direction_z,time,energy_log10,azimuth,zenith,event_no from truth")
counter2=0
Truth=np.zeros(11)
for row in Cursor2:
    counter2=counter2+1
    stack=[row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10]]
    Truth=np.vstack((Truth,stack))
    if counter2%10000==0:
        print (counter,Truth.shape)
        break
Truth=np.delete(Truth,[0],axis=0)
uniquesdata=np.unique(Data[:,5]
)
uniquesTruth=np.unique(Truth[:,10])
params= {
   "max_length" :10000,
    "features" : [ "x" , "y" , "z" , "time" , "charge_log10" ],
    "targets":["vertex_x","vertex_y","vertex_z"],
    "optimizer":"adam"
 }
loss="mse"
x=Data[:,0:5]
y=Truth[:,0:3]
model = tcn_model_26(
        params["max_length"], len(params["features"]), len(params["targets"])
    )
model.compile(optimizer=params["optimizer"], loss=loss)
model.fit(x,y,epochs=100,validation_split=0.2)