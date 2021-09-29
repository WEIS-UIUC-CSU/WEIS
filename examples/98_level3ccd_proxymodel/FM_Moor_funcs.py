# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 18:03:04 2021

@author: bayat2
"""
import numpy as np
from net_Target_funcs import net_Target1_rest_func, net_Target2_rest_func, net_Target1_Suspend_func, net_Target2_Suspend_func

def FM_Moor_Slack(t,qq,params):
    
    R11=params.R11; R12=params.R12
    R21=params.R21; R22=params.R22
    #k_add=params.k_add
    W=params.W
    r_AGs_1=params.r_AGs_1
    r_AGs_2=params.r_AGs_2
    r_EO_1=params.r_EO_1
    r_EO_2=params.r_EO_2
    
    rp1_O_1=qq[:,4]+R11*r_AGs_1[0]+R12*r_AGs_1[1]
    rp1_O_2=qq[:,5]+R21*r_AGs_1[0]+R22*r_AGs_1[1]
    
    rp2_O_1=qq[:,4]+R11*r_AGs_2[0]+R12*r_AGs_2[1]
    rp2_O_2=qq[:,5]+R21*r_AGs_2[0]+R22*r_AGs_2[1]
    
    l1=np.sqrt((rp1_O_1-r_EO_1[0])**2)
    l2=np.sqrt((rp2_O_1-r_EO_2[0])**2)
    
    h1=rp1_O_2-r_EO_1[1]
    h2=rp2_O_2-r_EO_2[1]
    
    l1=np.reshape(l1,(1,-1))
    l2=np.reshape(l2,(1,-1))
    h1=np.reshape(h1,(1,-1))
    h2=np.reshape(h2,(1,-1))

    V1_1=net_Target1_rest_func(np.r_[l1,h1])
    H1_1=net_Target2_rest_func(np.r_[l1,h1])
    
    V2_1=net_Target1_rest_func(np.r_[l2,h2])
    H2_1=net_Target2_rest_func(np.r_[l2,h2])

    V1_2=net_Target1_Suspend_func(np.r_[l1,h1])
    H1_2=net_Target2_Suspend_func(np.r_[l1,h1])
    
    V2_2=net_Target1_Suspend_func(np.r_[l2,h2])
    H2_2=net_Target2_Suspend_func(np.r_[l2,h2])

    indx1=np.abs(V1_1)>W
    indx2=np.abs(V2_1)>W
    
    V1=(~indx1)*V1_1+(indx1)*V1_2
    H1=(~indx1)*H1_1+(indx1)*H1_2

    V2=(~indx2)*V2_1+(indx2)*V2_2
    H2=(~indx2)*H2_1+(indx2)*H2_2
    
    V1=1.5*V1 # in 2D we have 2 mooring line,s but in 3D we have 3 mooring lines
    H1=1.5*H1
    V2=1.5*V2
    H2=1.5*H2
    
    beta_1=np.arctan2(0,rp1_O_1-r_EO_1[0])
    beta_2=np.arctan2(0,rp2_O_1-r_EO_2[0])

    T1=-np.r_[H1*np.cos(beta_1),V1]
    T2=-np.r_[H2*np.cos(beta_2),V2]
    
    T1_p_1=R11*T1[0,:].T+R21*T1[1,:].T
    T1_p_2=R12*T1[0,:].T+R22*T1[1,:].T
    T1_p=np.r_[np.reshape(T1_p_1,(1,-1)),np.reshape(T1_p_2,(1,-1))]
    
    T2_p_1=R11*T2[0,:].T+R21*T2[1,:].T
    T2_p_2=R12*T2[0,:].T+R22*T2[1,:].T
    T2_p=np.r_[np.reshape(T2_p_1,(1,-1)),np.reshape(T2_p_2,(1,-1))]

    M1_p=r_AGs_1[1]*T1_p[0,:]-r_AGs_1[0]*T1_p[1,:]
    M2_p=r_AGs_2[1]*T2_p[0,:]-r_AGs_2[0]*T2_p[1,:]

    F_Moor=T1_p+T2_p
    M_Moor=M1_p+M2_p #-k_add*[zeros(1,numel(t));zeros(1,numel(t));q(:,13).'];

    return F_Moor,M_Moor