# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:00:40 2021

@author: bayat2
"""

def compute_d(z,params):
    aa1=params.aa1;
    aa2=params.aa2;
    aa3=params.aa3;
    d1=params.d1;
    d2=params.d2;
    dtip=params.dtip;
    l=params.l;
    
    d=(d1)*(z<=aa1)+\
    (d1+(d2-d1)/aa2*(z-aa1))*((z>aa1) & (z<=aa1+aa2))+\
    (d2)*((z>aa1+aa2) & (z<=aa1+aa2+aa3))+\
    (d2+(dtip-d2)/l*(z-aa1-aa2-aa3))*(z>aa1+aa2+aa3);
    return d