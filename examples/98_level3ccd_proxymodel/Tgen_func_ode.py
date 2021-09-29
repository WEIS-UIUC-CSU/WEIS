#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:54:57 2021

@author: bayat2
"""
import numpy as np

def Tgen_func(t,q_scaled,blade_pitch,auxdata,omega_filter_last):
    k_star=2.165573001711226e+06
    A_scaled=auxdata.A_scaled;
    B_scaled=auxdata.B_scaled;
    
    #omega_filter_last=A_scaled[7]+B_scaled[7,7]*q_scaled[:,7]
    
    if blade_pitch>np.deg2rad(0.5*4):
        T_gen=min(5e6/0.944/omega_filter_last,47402.91*97)
    elif omega_filter_last<670*2*np.pi/60/97:   
        T_gen=0
    elif omega_filter_last<871*2*np.pi/60/97:
        T_gen=(k_star*(871*2*np.pi/60/97)**2-0)/(871*2*np.pi/60/97-670*2*np.pi/60/97)*(omega_filter_last-670*2*np.pi/60/97)
    elif omega_filter_last<1161.96*2*np.pi/60/97:
        T_gen=k_star*omega_filter_last**2
    elif omega_filter_last<1173.7*2*np.pi/60/97:
        T_gen=(43093.55*97-k_star*(1161.96*2*np.pi/60/97)**2)/(1173.7*2*np.pi/60/97-1161.96*2*np.pi/60/97)*(omega_filter_last-1161.96*2*np.pi/60/97)+k_star*(1161.96*2*np.pi/60/97)**2
    else:
        T_gen=5e6/0.944/omega_filter_last
        
    return T_gen
            
        