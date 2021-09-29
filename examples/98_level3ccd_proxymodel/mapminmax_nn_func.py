# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 13:52:13 2021

@author: bayat2
"""
import numpy as np

def mapminmax_apply(x,x_step1_xoffset,x_step1_gain,x_step1_ymin):
    y=x-x_step1_xoffset
    y=y*x_step1_gain
    y=y+x_step1_ymin
    return y

def tansig_apply(n):
    a = 2 / (1 + np.exp(-2*n)) - 1
    return a

def mapminmax_reverse(y,y_step1_xoffset,y_step1_gain,y_step1_ymin):
    x=y-y_step1_ymin
    x=x/y_step1_gain
    x=x+y_step1_xoffset
    return x