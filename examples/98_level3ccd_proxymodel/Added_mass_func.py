# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:46:37 2021

@author: bayat2
"""
import numpy as np

def Added_mass_diff_sections(t,qq,params):
    Vd=params.Vd
    acv=params.acv
    rb=params.rb
    L_cf=params.L_cf
    I_add=params.I_add

    Ca=params.Ca
    d1=params.d1
    rho_w=params.rho_w

    A11=Ca*rho_w*Vd
    A22=A11
    A33=Ca*1/12*rho_w*np.pi*(d1)**3*np.ones(len(t))
    A15=A11*(acv-rb)
    A24=-A15
    A44=Ca*I_add
    A55=A44

    A=np.zeros((len(t),3,3))
    C_A=np.zeros((len(t),3,3))

    for i in np.arange(len(t)):
        A[i,:,:]=np.array([[A11[i],0,A15[i]],\
                           [ 0,  A33[i],  0],\
                               [ A15[i],0 ,A55[i]]])
    
        C_A[i,:,:]=np.array([[0,0, A11[i]*qq[i,1]],\
                           [ 0,  0,  -A11[i]*qq[i,0]],\
                           [ -A11[i]*qq[i,1],A11[i]*qq[i,0] ,0]])
    
    return A,C_A
    