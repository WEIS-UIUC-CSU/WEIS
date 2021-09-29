# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:46:52 2021

@author: bayat2
"""

#export LD_LIBRARY_PATH=/home/bayat2/ipopt/lib:$LD_LIBRARY_PATH
import os
import random, string
import numpy as np
from auxdata_PID_gen_class import auxdata
from matplotlib.pyplot import plot, pause
import matplotlib.pyplot as plt
import openmdao.api as om
from openmdao.utils.general_utils import set_pyoptsparse_opt
import dymos as dm
from v_wind_func import v_windb_func_rigid
from U_rel_func import U_rel_func
from Hydrostatic_func import Hydrosttaic_diff_sections_rigid
from Added_mass_func import Added_mass_diff_sections
from FM_Moor_funcs import FM_Moor_Slack
from FM_Wave_funcs import FM_Wave_diff_sections_trapz
from wave_profile_func import wave_profile_func
import matplotlib
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('x-large')
from WT_Continous_PID_func import WT_Continous_PID
from Tgen_func_ode import Tgen_func


def obj_inner_PID(x,wind_avg,wind_type):
    
    auxdata_instance=auxdata(x,wind_avg,wind_type)

    t0 = 0     # [s] 
    tf=100*1   # finale simulation time [s]
    
    h_t=0.01
    n_t=int(tf/h_t+1)
    t=np.linspace(0,tf,n_t)

    x0_scaled=np.array([0,0,0,2*1.2671/auxdata_instance.omega_max-1,0,0,0,2*1.2671/auxdata_instance.omega_max-1,-1])
    x_scaled=np.zeros((n_t,9))
    x_scaled[0,:]=x0_scaled
    
    #%% Run PID
    """--------------------------Run PID-----------------------"""
    U1=np.zeros(n_t)
    U2=np.zeros(n_t)
    omega_filter_last=1.2671
    platform_pitch_last=0
    for i in np.arange(1,n_t):
        k1=WT_Continous_PID(np.array([t[i-1]]),np.array([x_scaled[i-1,:]]),auxdata_instance,U2[i-1],omega_filter_last,platform_pitch_last)
        #k2=WT_Continous_PID(np.array([(t[i-1]+t[i])/2]),np.array([x_scaled[i-1,:]])+h_t*k1/2,auxdata_instance,U2[i-1],omega_filter_last,platform_pitch_last)
        #k3=WT_Continous_PID(np.array([(t[i-1]+t[i])/2]),np.array([x_scaled[i-1,:]])+h_t*k2/2,auxdata_instance,U2[i-1],omega_filter_last,platform_pitch_last)
        #k4=WT_Continous_PID(np.array([t[i]]),np.array([x_scaled[i-1,:]])+h_t*k3,auxdata_instance,U2[i-1],omega_filter_last,platform_pitch_last)
        #x_scaled[i,:]=x_scaled[i-1,:]+h_t/6*(k1+2*k2+2*k3+k4)
        x_scaled[i,:]=x_scaled[i-1,:]+h_t*k1
    
        A_scaled=auxdata_instance.A_scaled
        B_scaled=auxdata_instance.B_scaled
    
        qq=np.zeros((1,9))
        for j in np.arange(9):
            qq[:,j]=A_scaled[j]+B_scaled[j,j]*x_scaled[i,j]
         
        if qq[:,8]<0:
            qq[:,8]=0
            x_scaled[i,8]=-1
        elif qq[:,8]>np.deg2rad(39):
            qq[:,8]=np.deg2rad(39)
            x_scaled[i,8]=1
        
        U2[i]= auxdata_instance.Kp*1/(U2[i-1]/np.deg2rad(6.302336)+1)*auxdata_instance.N_Gear*(omega_filter_last-auxdata_instance.omega_r)+qq[:,8]+auxdata_instance.K_feedback_pitch*platform_pitch_last**2
        if U2[i]<0:
            U2[i]=0
        elif U2[i]>np.deg2rad(39):
            U2[i]=np.deg2rad(39)
        
        U1[i]=Tgen_func(t[i],np.array([x_scaled[i,:]]),U2[i],auxdata_instance,omega_filter_last)
    
        omega_filter_last=A_scaled[7]+B_scaled[7,7]*x_scaled[i,7]
        platform_pitch_last=A_scaled[6]+B_scaled[6,6]*x_scaled[i,6];

    #%% Results    
    """--------------------------Results-----------------------"""
    q_vec=np.zeros((n_t,9))
    for i in np.arange(np.shape(x_scaled)[1]):
        q_vec[:,i]=auxdata_instance.A_scaled[i]+auxdata_instance.B_scaled[i,i]*x_scaled[:,i]

    theta=q_vec[:,6]
    
    R11=np.cos(theta)
    R12=np.sin(theta)
    R21=-np.sin(theta)
    R22=np.cos(theta)

    v_wind=np.r_[np.reshape(auxdata_instance.v(t),(1,-1)),0*np.reshape(auxdata_instance.v(t),(1,-1))]

    v_wind_b_1=R11*v_wind[0,:]+R21*v_wind[1,:]
    v_wind_b_2=R12*v_wind[0,:]+R22*v_wind[1,:]

    v_wind_b=np.r_[np.reshape(v_wind_b_1,(1,-1)),np.reshape(v_wind_b_2,(1,-1))]

    v_r_1=q_vec[:,0]+auxdata_instance.Dr*q_vec[:,2]

    U_rel=v_wind_b[0,:]-v_r_1

    lambda_R=auxdata_instance.R_rotor*q_vec[:,7]/U_rel

    Cp_vec=auxdata_instance.Cpfunc.ev(lambda_R,U2)

    Ct_vec=auxdata_instance.Ctfunc.ev(lambda_R,U2)

    P_a=Cp_vec*0.5*auxdata_instance.rho_air*np.pi*auxdata_instance.R_rotor**2*(U_rel)**3

    P_u=q_vec[:,7]*U1*auxdata_instance.etha

    taw_a=1/2*auxdata_instance.rho_air*np.pi*auxdata_instance.R_rotor**3*Cp_vec/lambda_R*U_rel**2

    taw_g=U1

    omega_dot=1/auxdata_instance.J_rotor*(taw_a-taw_g)

    P_inter=auxdata_instance.J_rotor*(q_vec[:,3])*omega_dot*auxdata_instance.etha


    Tg_dot=np.zeros(n_t)
    Tg_dot[1:-1]=np.diff(U1[1:])/np.diff(t[1:])
    Tg_dot[-1]=Tg_dot[-2]
    Tg_mean_abs=np.abs(np.trapz(Tg_dot,t)/tf)*20

    for i in np.arange(n_t):
        if np.abs(Tg_dot[i])>Tg_mean_abs:
            Tg_dot[i]=Tg_dot[i-1]

    theta_b_dot=np.zeros(n_t)
    theta_b_dot[1:-1]=np.diff(U2[1:])/np.diff(t[1:])
    theta_b_dot[-1]=theta_b_dot[-2]
    theta_b_mean_abs=np.abs(np.trapz(theta_b_dot,t)/tf)*20

    for i in np.arange(n_t):
        if np.abs(theta_b_dot[i])>theta_b_mean_abs:
            theta_b_dot[i]=theta_b_dot[i-1]

    obj_integrand=-auxdata_instance.etha*(P_a-1e-7*Tg_dot**2-1e7*theta_b_dot**2)/tf
    objective_inner=np.trapz(obj_integrand,t)
    
    return objective_inner