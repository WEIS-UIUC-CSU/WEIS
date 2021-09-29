# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:07:59 2021

@author: bayat2
"""
import numpy as np
from compute_d import compute_d

def Hydrosttaic_diff_sections_rigid(t,qq,params):
    R11=params.R11; R12=params.R12
    R21=params.R21; R22=params.R22
    Lc=params.Lc
    rb=params.rb
    rho_w=params.rho_w
    g=params.g
    Hw=params.Hw
    nPoints=params.nPoints    #number of points used in numerial quadrature

    L_cf=(Lc-rb-qq[:,5]+Hw*1)/R22+rb
    
    start_points=np.zeros(len(L_cf)) # starting point
    start_points=np.reshape(start_points, (-1, 1))
    
    end_points=L_cf
    end_points=np.reshape(end_points, (-1, 1))
    
    x_lin=np.linspace(0,1,nPoints)
    x_lin=np.reshape(x_lin, (1, -1))
    
    z=start_points+(end_points-start_points)@x_lin
    z_t=z.T
    
    z_total_vec=z_t.T.flatten()
    d=compute_d(z_total_vec,params)
    Ac=np.pi/4*d**2
    
    Ac_mat=np.reshape(Ac,(nPoints,len(t)),order='F')
    
    Vd=np.zeros(len(L_cf))
    acv=np.zeros(len(L_cf))
    Ic=np.reshape(np.pi/64*d**4,(nPoints,len(t)),order='F')
    Ibb=np.zeros(len(L_cf))
    
    for i in np.arange(len(L_cf)):
        Vd[i]=np.trapz(Ac_mat[:,i],z_t[:,i])
        acv[i]=np.trapz(z_t[:,i]*Ac_mat[:,i],z_t[:,i])/Vd[i]
        Ibb[i]=rho_w*np.trapz(Ic[:,i]+z_t[:,i]**2*Ac_mat[:,i],z_t[:,i])

    I_add=Ibb-rho_w*Vd*rb*(2*acv-rb)
    
    params.Vd=Vd
    params.acv=acv
    
    F_hsI=rho_w*g*Vd

    F_hs_1=R11*0+R21*F_hsI
    F_hs_2=R12*0+R22*F_hsI
    
    F_hs=np.c_[F_hs_1, F_hs_2].T

    theta=qq[:,6]

    dwp=compute_d(L_cf,params);

    Iwp=np.pi/64*dwp**4
    
    Mwp=-rho_w*g*Iwp*(np.sin(theta)+1/2*np.sin(theta)*(np.tan(theta))**2)

    rC=np.c_[np.zeros(len(t)), acv-rb].T  #p coordinate
    M_hs=rC[1,:]*F_hs[0,:]+Mwp

    params.L_cf=L_cf;
    params.I_add=I_add;
    params.d_mat=np.reshape(d,(nPoints,len(t)),order='F');
    params.Ac_mat=Ac_mat;
    params.z=z;

    return F_hs,M_hs,params