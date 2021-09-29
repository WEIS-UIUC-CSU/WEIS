# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:48:16 2021

@author: bayat2
"""

import numpy as np

def v_windb_func_rigid(qq,params):
    l=params.l
    v_wind=params.v_wind

    R11=params.R11; R12=params.R12
    R21=params.R21; R22=params.R22
    
    v_wind_b_1=R11*v_wind[0,:]+R21*v_wind[1,:]
    v_wind_b_2=R12*v_wind[0,:]+R22*v_wind[1,:]

    v_wind_b=np.c_[v_wind_b_1, v_wind_b_2].T
    return v_wind_b
    