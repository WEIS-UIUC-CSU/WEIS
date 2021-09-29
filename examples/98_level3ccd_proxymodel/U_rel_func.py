# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 20:05:05 2021

@author: bayat2
"""
import numpy as np

def U_rel_func(qq,params):
    v_wind_b=params.v_wind_b;
    Dr=params.Dr;

    v_r_1=qq[:,0]+Dr*qq[:,2] #rotor velocity in platform x coordinate

    U_rel=[v_wind_b[0,:]-v_r_1]
    U_rel_pos=U_rel; 
    return U_rel_pos