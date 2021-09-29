# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 00:15:34 2021

@author: bayat2
"""
import numpy as np

def max_tension(z,params):
    a1=params.a1
    a2=params.a2
    a3=params.a3
    a4=params.a4
    l=params.l

    Do=a3-a4*z;
    Di=Do-2*(a1-a2*z);
    f_min=np.abs(Do*(l-z)/(Do**4-Di**4));
    return f_min