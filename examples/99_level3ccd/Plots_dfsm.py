#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:07:52 2021

@author: bayat2
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('x-large')

with open('output/dfsm.pkl','rb') as dfsm_file:
    dfsm=pkl.load(dfsm_file)
    
wind_speed_vec=np.linspace(3.,25.,1570)  
hub_height_vec=np.linspace(145.0, 155.0,1570) 
tower_top_diameter_vec=np.linspace(5.9, 7.1,1570)  
tower_bottom_thickness_vec=np.linspace(0.035, 0.045,1570)  
tower_top_thickness_vec=np.linspace(0.015, 0.025,1570)  
rotor_diameter_vec=np.linspace(230.0, 250.0,1570)   

#%%
"""------------------------Plot Controls---------------------"""
U_OPS_predict=dfsm.predict_sm_U_OPS(np.c_[wind_speed_vec,hub_height_vec,tower_top_diameter_vec,tower_bottom_thickness_vec,tower_top_thickness_vec,rotor_diameter_vec])
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_U_OPS[:,0],'o', linewidth=2,label='Samples')
ax.plot(U_OPS_predict[:,0],'o', linewidth=2,label='Trained')
ax.set_ylabel('$V \, \mathrm{[m/s]\,\,(Wind\,\, Speed)}$',fontsize=15)
ax.legend(prop = fontP) 
ax.set_title('Controls')   
    
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(1e-6*dfsm.F_TRAIN_U_OPS[:,1],'o', linewidth=1,label='Samples')
ax.plot(1e-6*U_OPS_predict[:,1],'o', linewidth=1,label='Trained')
ax.set_ylabel('$T_{gen} \, \mathrm{[MNm]\,\,(Generator\,\, Torque)}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Controls')  

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_U_OPS[:,2],'o', linewidth=2,label='Samples')
ax.plot(U_OPS_predict[:,2],'o', linewidth=2,label='Trained')
ax.set_ylabel(r'$\beta \, \mathrm{[rad]\,\,(Blade\,\, Pitch)}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Controls')  

#%%
"""------------------------Plot States---------------------"""
X_OPS_predict=dfsm.predict_sm_X_OPS(np.c_[wind_speed_vec,hub_height_vec,tower_top_diameter_vec,tower_bottom_thickness_vec,tower_top_thickness_vec,rotor_diameter_vec])

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_X_OPS[:,0],'o', linewidth=2,label='Samples')
ax.plot(X_OPS_predict[:,0],'o', linewidth=2,label='Trained')
ax.set_ylabel('$x \, \mathrm{[m]\,\,(fore-aft)}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('States')  

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_X_OPS[:,1],'o', linewidth=2,label='Samples')
ax.plot(X_OPS_predict[:,1],'o', linewidth=2,label='Trained')
ax.set_ylabel('$\dot{x} \, \mathrm{[m/s]\,\,(fore-aft\,\, speed)}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('States') 

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_X_OPS[:,2],'o', linewidth=2,label='Samples')
ax.plot(X_OPS_predict[:,2],'o', linewidth=2,label='Trained')
ax.set_ylabel(r'$\omega \, \mathrm{[rad/s]\,\,(rotor\,\, rotational\,\,speed)}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('States') 
#%%
"""------------------------Plot Outputs---------------------"""
Y_OPS_predict=dfsm.predict_sm_Y_OPS(np.c_[wind_speed_vec,hub_height_vec,tower_top_diameter_vec,tower_bottom_thickness_vec,tower_top_thickness_vec,rotor_diameter_vec])

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_Y_OPS[:,0],'o', linewidth=2,label='Samples')
ax.plot(Y_OPS_predict[:,0],'o', linewidth=2,label='Trained')
ax.set_ylabel('$V \, \mathrm{[m/s]\,\,(Wind\,\, Speed)}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Outputs') 

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_Y_OPS[:,1],'o', linewidth=2,label='Samples')
ax.plot(Y_OPS_predict[:,1],'o', linewidth=2,label='Trained')
ax.set_ylabel('$P_{gen} \, \mathrm{[kW]\,\,(Generator\,\, Power)}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Outputs') 

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_Y_OPS[:,2],'o', linewidth=2,label='Samples-GenTorq')
ax.plot(dfsm.F_TRAIN_Y_OPS[:,6],'o', linewidth=2,label='Samples-RotTorq')
ax.plot(Y_OPS_predict[:,2],'o', linewidth=2,label='Trained-GenTorq')
ax.plot(Y_OPS_predict[:,6],'o', linewidth=2,label='Trained-RotTorq')
ax.set_ylabel('$Torque \, \mathrm{[kN-m]}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Outputs')

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_Y_OPS[:,3],'o', linewidth=2,label='Samples-GenSpeed')
ax.plot(dfsm.F_TRAIN_Y_OPS[:,4],'o', linewidth=1,label='Samples-RotSpeed')
ax.plot(Y_OPS_predict[:,3],'o', linewidth=2,label='Trained-GenSpeed')
ax.plot(Y_OPS_predict[:,4],'o', linewidth=2,label='Trained-RotSpeed')
ax.set_ylabel('$speed \, \mathrm{[rad/s]}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Outputs')

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_Y_OPS[:,5],'o', linewidth=2,label='Samples')
ax.plot(Y_OPS_predict[:,5],'o', linewidth=2,label='Trained')
ax.set_ylabel('$RotThrust \, \mathrm{[kN]\,\,(Rotor\,\, thrust)}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Outputs')

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_Y_OPS[:,7],'o', linewidth=2,label='Samples-TwrBsFxt')
ax.plot(dfsm.F_TRAIN_Y_OPS[:,8],'o', linewidth=1,label='Samples-TwrBsFyt')
ax.plot(dfsm.F_TRAIN_Y_OPS[:,9],'o', linewidth=1,label='Samples-TwrBsFzt')
ax.plot(Y_OPS_predict[:,7],'o', linewidth=2,label='Trained-TwrBsFxt')
ax.plot(Y_OPS_predict[:,8],'o', linewidth=2,label='Trained-TwrBsFyt')
ax.plot(Y_OPS_predict[:,9],'o', linewidth=2,label='Trained-TwrBsFzt')
ax.set_ylabel('$Tower\, \,base \,forces \mathrm{[kN]}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Outputs')

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_Y_OPS[:,10],'o', linewidth=2,label='Samples-TwrBsMxt')
ax.plot(dfsm.F_TRAIN_Y_OPS[:,11],'o', linewidth=1,label='Samples-TwrBsMyt')
ax.plot(dfsm.F_TRAIN_Y_OPS[:,12],'o', linewidth=1,label='Samples-TwrBsMzt')
ax.plot(Y_OPS_predict[:,10],'o', linewidth=2,label='Trained-TwrBsMxt')
ax.plot(Y_OPS_predict[:,11],'o', linewidth=2,label='Trained-TwrBsMyt')
ax.plot(Y_OPS_predict[:,12],'o', linewidth=2,label='Trained-TwrBsMzt')
ax.set_ylabel('$Tower\, \,base \,moments \mathrm{[kN-m]}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Outputs')

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_Y_OPS[:,13],'o', linewidth=2,label='Samples-YawBrFxp')
ax.plot(dfsm.F_TRAIN_Y_OPS[:,14],'o', linewidth=1,label='Samples-YawBrFyp')
ax.plot(dfsm.F_TRAIN_Y_OPS[:,15],'o', linewidth=1,label='Samples-YawBrFzp')
ax.plot(Y_OPS_predict[:,13],'o', linewidth=2,label='Trained-YawBrFxp')
ax.plot(Y_OPS_predict[:,14],'o', linewidth=2,label='Trained-YawBrFyp')
ax.plot(Y_OPS_predict[:,15],'o', linewidth=2,label='Trained-YawBrFzp')
ax.set_ylabel('$Tower\, \,top \,forces \mathrm{[kN]}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Outputs')

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(dfsm.F_TRAIN_Y_OPS[:,16],'o', linewidth=2,label='Samples-YawBrMxp')
ax.plot(dfsm.F_TRAIN_Y_OPS[:,17],'o', linewidth=1,label='Samples-YawBrMyp')
ax.plot(dfsm.F_TRAIN_Y_OPS[:,18],'o', linewidth=1,label='Samples-YawBrMzp')
ax.plot(Y_OPS_predict[:,16],'o', linewidth=2,label='Trained-YawBrMxp')
ax.plot(Y_OPS_predict[:,17],'o', linewidth=2,label='Trained-YawBrMyp')
ax.plot(Y_OPS_predict[:,18],'o', linewidth=2,label='Trained-YawBrMzp')
ax.set_ylabel('$Tower\, \,top \,moments \mathrm{[kN]}$',fontsize=15)
ax.legend(prop = fontP)
ax.set_title('Outputs')