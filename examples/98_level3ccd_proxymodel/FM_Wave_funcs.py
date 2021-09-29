# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 18:59:34 2021

@author: bayat2
"""
import numpy as np

def FM_Wave_diff_sections_trapz(t,qq,params):
    Ca=params.Ca
    CD=params.CD   
    R11=params.R11; R12=params.R12
    R21=params.R21; R22=params.R22
    rho_w=params.rho_w
    vf=params.vf
    vf_dot=params.vf_dot
    l=params.l
    rb=params.rb
    L_cf=params.L_cf
    nPoints=params.nPoints #number of points in each row
    Dcs=params.d_mat
    Acs=params.Ac_mat
    z=params.z-rb
    
    v_fp_1=R11*vf[0,:]+R21*vf[1,:]
    v_fp_2=R12*vf[0,:]+R22*vf[1,:]
    v_fp=np.r_[np.reshape(v_fp_1,(1,-1)),np.reshape(v_fp_2,(1,-1))]
    
    v_dot_fp_1=R11*vf_dot[0,:]+R21*vf_dot[1,:]
    v_dot_fp_2=R12*vf_dot[0,:]+R22*vf_dot[1,:]
    v_dot_fp=np.r_[np.reshape(v_dot_fp_1,(1,-1)),np.reshape(v_dot_fp_2,(1,-1))]
    
    v_dot_fp_normal=np.r_[np.reshape(v_dot_fp[0,:],(1,-1)),np.zeros((1,len(t)))]
    v_each_segment=np.zeros((np.shape(z)[1],2,np.shape(z)[0]))
    
    for j in np.arange(np.shape(z)[0]): #time 
        v_each_segment[:,0,j]=qq[j,0]+qq[j,2]*z[j,:] #vx+z w_y
        v_each_segment[:,1,j]=qq[j,1]  #vz
        
  
    v_fp_mat=np.tile(v_fp,(nPoints,1,1))
    v_rel=v_fp_mat-v_each_segment;
    v_rel_normal=v_rel
    v_rel_normal[:,1,:]=0
    
    FM_Wave=np.zeros((len(t),3))
    
    for j in np.arange(len(t)):
        df1=rho_w*(1+Ca)*(Acs[:,j]*v_dot_fp_normal[0,j])+1/2*rho_w*CD*((Dcs[:,j]*v_rel_normal[:,0,j])*np.abs(v_rel_normal[:,0,j]))
        df2=rho_w*(1+Ca)*(Acs[:,j]*v_dot_fp_normal[1,j])+1/2*rho_w*CD*((Dcs[:,j]*v_rel_normal[:,1,j])*np.abs(v_rel_normal[:,1,j]))
        dM1=(df1*z[j,:]) #df1*z
       
        FM_Wave[j,0]=np.trapz(df1,z[j,:])
        FM_Wave[j,1]=np.trapz(df2,z[j,:])
        FM_Wave[j,2]=np.trapz(dM1,z[j,:])
    
    FM_Wave=FM_Wave.T
    
    F_wave=FM_Wave[0:2,:]
    M_wave=FM_Wave[2:3,:]
    return F_wave,M_wave