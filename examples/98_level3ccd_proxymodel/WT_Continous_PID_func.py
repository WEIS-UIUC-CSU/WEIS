#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:07:20 2021

@author: bayat2
"""

import numpy as np
from v_wind_func import v_windb_func_rigid
from U_rel_func import U_rel_func
from Hydrostatic_func import Hydrosttaic_diff_sections_rigid
from Added_mass_func import Added_mass_diff_sections
from FM_Moor_funcs import FM_Moor_Slack
from FM_Wave_funcs import FM_Wave_diff_sections_trapz
from wave_profile_func import wave_profile_func
from Tgen_func_ode import Tgen_func

def WT_Continous_PID(t,q_scaled,auxdata_instance,U2_last,omega_filter_last,platform_pitch_last):
    
    etha=auxdata_instance.etha;
    
    #scaled data
    A_scaled=auxdata_instance.A_scaled
    B_scaled=auxdata_instance.B_scaled
    C_scaled=auxdata_instance.C_scaled
    D_scaled=auxdata_instance.D_scaled
    
    qq=q_scaled;
    for i in np.arange(q_scaled.shape[1]):
        qq[:,i]=A_scaled[i]+B_scaled[i,i]*q_scaled[:,i]
            
    #
    theta=qq[:,6]
    #Rotation matrix between inertial frame and platform frame
    #Inertail frame=[R11  R12;R21   R22]*Platform Frame
    R11=np.cos(theta);
    R12=np.sin(theta);
    R21=-np.sin(theta);
    R22=np.cos(theta);
    
    auxdata_instance.R11=R11; auxdata_instance.R12=R12 
    auxdata_instance.R21=R21; auxdata_instance.R22=R22
    
    #wave velocity and acceleration in Inertial Frame
    vf=np.zeros((2,len(t)))      # wave velocity
    vf_dot=np.zeros((2,len(t)))  # wave acceleration
    Hw=np.zeros(len(t))          # wave amplitude
    
    """
    for i in np.arange(len(t)):
        [vf_i,vf_dot_i,Hw_i] = wave_profile_func(t[i],qq[i,:],auxdata_instance)
        vf[0,i]=vf_i[0][0]
        vf[1,i]=vf_i[1][0]
        vf_dot[0,i]=vf_dot_i[0][0]
        vf_dot[1,i]=vf_dot_i[1][0]
        Hw[i]=Hw_i
    """
    auxdata_instance.vf=vf;
    auxdata_instance.vf_dot=vf_dot;
    
    # Wind velocity in Inertial coordinate
    wind_speed=auxdata_instance.v;
    v_wind=np.c_[wind_speed(t), 0*wind_speed(t)].T
    auxdata_instance.v_wind=v_wind;
    
    # rotor, generator torque, and pitch dynamics
    rho_air=auxdata_instance.rho_air;
    R_rotor=auxdata_instance.R_rotor;
    Cp=auxdata_instance.Cpfunc
    Ct=auxdata_instance.Ctfunc
    
    #wind velocity in body coordinate
    v_wind_b= v_windb_func_rigid(qq,auxdata_instance)
    auxdata_instance.v_wind_b=v_wind_b;
        
    [U_rel] = U_rel_func(qq,auxdata_instance)
    
    dq8=0.25*(qq[:,3]-qq[:,7])
    #platform_pitch=A_scaled[6]+B_scaled[6,6]*q_scaled[:,6];
    dq9=auxdata_instance.KI*1/(U2_last/np.deg2rad(6.302336)+1)*auxdata_instance.N_Gear*(omega_filter_last-auxdata_instance.omega_r)
    
    
    if qq[:,8]<0:
        qq[:,8]=0
    elif qq[:,8]>np.deg2rad(39):
        qq[:,8]=np.deg2rad(39)
    
    U2= auxdata_instance.Kp*1/(U2_last/np.deg2rad(6.302336)+1)*auxdata_instance.N_Gear*(omega_filter_last-auxdata_instance.omega_r)+qq[:,8]+auxdata_instance.K_feedback_pitch*platform_pitch_last**2
        
    if U2<0:
        U2=0
    elif U2>np.deg2rad(39):
        U2=np.deg2rad(39)
       
    U1=Tgen_func(t,q_scaled,U2,auxdata_instance,omega_filter_last)
    
    lambda_R=R_rotor*qq[:,7]/(U_rel)
    Cp_vec=Cp.ev(lambda_R,U2)
    Ct_vec=Ct.ev(lambda_R,U2)
    #Cp_vec=np.linspace(0,0.45,len(t))
    #Ct_vec=np.linspace(0,0.9,len(t))
        
    tau_a=1/2*rho_air*np.pi*R_rotor**3*Cp_vec/lambda_R*U_rel**2 #aerodynamic torque
    tau_g=U1 #generator torque
    T_a=Ct_vec*1/2*rho_air*np.pi*R_rotor**2*U_rel**2 #thrust
    F_aero=np.c_[T_a,np.zeros(len(t))].T #Force
    M_aero=T_a*auxdata_instance.Dr     #Moment
        
    [F_hs,M_hs,auxdata_instance_update] = Hydrosttaic_diff_sections_rigid(t,qq,auxdata_instance); 
    auxdata_instance.Vd=auxdata_instance_update.Vd
    auxdata_instance.acv=auxdata_instance_update.acv
    auxdata_instance.L_cf=auxdata_instance_update.L_cf
    auxdata_instance.I_add=auxdata_instance_update.I_add
    auxdata_instance.d_mat=auxdata_instance_update.d_mat
    auxdata_instance.Ac_mat=auxdata_instance_update.Ac_mat
    auxdata_instance.z=auxdata_instance_update.z
        
    [A,C_A] = Added_mass_diff_sections(t,qq,auxdata_instance);  
    
    [F_Moor,M_Moor] = FM_Moor_Slack(t,qq,auxdata_instance) #Slack
        
    [F_wave1,M_wave1] = FM_Wave_diff_sections_trapz(t,qq,auxdata_instance)
        
    B_addition=auxdata_instance.B_addition
    Addition_damping=np.zeros((6,len(t)))
    Addition_damping[0,:]=-B_addition[0,0]*qq[:,0] #experimental additional damping in surge
    Addition_damping[1,:]=-B_addition[2,2]*qq[:,1] #experimental additional damping in heave
    F_wave=F_wave1+Addition_damping[0:2,:]
    M_wave=M_wave1
        
    F=(F_hs+F_Moor+F_wave+F_aero)
    M=(M_hs+M_Moor+M_wave+M_aero)
        
    #Right hand side - part 1 equations : eq 2.19
    RHS_1_1=auxdata_instance.g*auxdata_instance.mT*np.sin(qq[:,6])
    RHS_1_2=-auxdata_instance.g*auxdata_instance.mT*np.cos(qq[:,6])
    RHS_1_3=auxdata_instance.g*(np.sin(qq[:,6])*(auxdata_instance.Dr*(auxdata_instance.mnc+auxdata_instance.mr)+auxdata_instance.Dt*auxdata_instance.mt)+\
                   np.cos(qq[:,6])*(auxdata_instance.dnc*auxdata_instance.mnc-auxdata_instance.dr*auxdata_instance.mr))# NEGATIVE
    RHS_1_4=np.zeros((len(t),1))
    RHS_1=np.r_[np.reshape(RHS_1_1,(1,-1)), np.reshape(RHS_1_2,(1,-1)), np.reshape(RHS_1_3,(1,-1)), RHS_1_4.T]

    #Right hand side part 2 equations : eq 2.19
    RHS_2=np.r_[F,M,np.reshape(tau_a-tau_g,(1,-1))]
        
    #writing eq 2.19 in vector format (this needs explanation!)
    a11=auxdata_instance.M_sys[0,0]+A[:,0,0]
    a15=auxdata_instance.M_sys[0,2]+A[:,0,2]
    a26=auxdata_instance.M_sys[1,2]
    a33=auxdata_instance.M_sys[1,1]+A[:,1,1]
    a55=auxdata_instance.M_sys[2,2]+A[:,2,2]
    a77=auxdata_instance.M_sys[3,3]
        
    b11=0+C_A[:,0,0]
    b12=auxdata_instance.mT*qq[:,2]+C_A[:,0,1]
    b13=-auxdata_instance.M26*qq[:,2]+C_A[:,0,2]
    b14=0;
    
    b21=-auxdata_instance.mT*qq[:,2]+C_A[:,1,0]
    b22=0+C_A[:,1,1]
    b23=-auxdata_instance.M15*qq[:,2]+C_A[:,1,2]
    b24=0;
    
    b31=auxdata_instance.mT*qq[:,1]+C_A[:,2,0]
    b32=-auxdata_instance.mT*qq[:,0]+C_A[:,2,1]
    b33=auxdata_instance.M26*qq[:,0]+ auxdata_instance.M15*qq[:,1]+C_A[:,2,2]
    b34=0

    b41=0
    b42=0
    b43=0
    b44=0
        
    det_a=a33*a15**2 + a11*a26**2 - a11*a33*a55
    n11=1/det_a*(a26**2 - a33*a55)
    n12=1/det_a*(a15*a26)
    n13=1/det_a*(a15*a33)
    n14=0
    n21=1/det_a*(a15*a26)
    n22=1/det_a*(a15**2 - a11*a55)
    n23=1/det_a*(-a11*a26)
    n24=0
    n31=1/det_a*(a15*a33)
    n32=1/det_a*(-a11*a26)
    n33=1/det_a*(-a11*a33)
    n34=0
    n41=0
    n42=0
    n43=0
    n44=1/a77

    int1=b11*qq[:,0]+b12*qq[:,1]+\
    b13*qq[:,2]#+b14.*q(:,4);
    int2=b21*qq[:,0]+b22*qq[:,1]+\
    b23*qq[:,2]#+b24.*q(:,4); 
    int3=b31*qq[:,0]+b32*qq[:,1]+\
    b33*qq[:,2]#+b34.*q(:,4); 
    int4=0#;%b41.*q(:,1)+b42.*q(:,2)+...
    #b43.*q(:,3)+b44.*q(:,4);
        
    p_1=RHS_1[0,:]+RHS_2[0,:]-int1
    p_2=RHS_1[1,:]+RHS_2[1,:]-int2
    p_3=RHS_1[2,:]+RHS_2[2,:]-int3
    #%p_4=RHS_1(4,:).'+RHS_2(4,:).'-int4;
    p_4=RHS_2[3,:]
        
    dq1=n11*p_1+n12*p_2+\
    n13*p_3;#+n14.*p_4;
    dq2=n21*p_1+n22*p_2+\
    n23*p_3;#+n24.*p_4;
    dq3=n31*p_1+n32*p_2+\
    n33*p_3;#+n34.*p_4;
    dq4=n44*p_4#%+n41.*p_1+...
    #n42.*p_2+n43.*p_3;  

    dq5=R11*qq[:,0]+R12*qq[:,1]
    dq6=R21*qq[:,0]+R22*qq[:,1]
    dq7=qq[:,2]
        
    Dq_scaled1=B_scaled[0,0]**-1*dq1
    Dq_scaled2=B_scaled[1,1]**-1*dq2
    Dq_scaled3=B_scaled[2,2]**-1*dq3
    Dq_scaled4=B_scaled[3,3]**-1*dq4
    Dq_scaled5=B_scaled[4,4]**-1*dq5
    Dq_scaled6=B_scaled[5,5]**-1*dq6
    Dq_scaled7=B_scaled[6,6]**-1*dq7
    Dq_scaled8=(B_scaled[7,7]**-1*dq8)*1
    Dq_scaled9=(B_scaled[8,8]**-1*dq9)*1
    
    l=auxdata_instance.l
    max_ten_z_buttom=auxdata_instance.max_ten_z_buttom
    Do_max_ten=auxdata_instance.Do_max_ten
    Di_max_ten=auxdata_instance.Di_max_ten
        
    tansion_at_tower_root=T_a*(l-max_ten_z_buttom)*(Do_max_ten/2)/(np.pi/64*(Do_max_ten**4-Di_max_ten**4))*1e-6 #Mpa
        
    P_a=Cp_vec*0.5*rho_air*np.pi*R_rotor**2*U_rel**3
    P_max=auxdata_instance.P_max
        
    objective=-1e-9*(P_a-1e-7*U1**2-1e7*U2**2)*etha
        
    path_con_1=tansion_at_tower_root
    path_con_2=tau_g*qq[:,3]*etha/P_max
    
    q_scaled_dot=np.c_[Dq_scaled1,Dq_scaled2,Dq_scaled3,
          Dq_scaled4,Dq_scaled5,Dq_scaled6,
          Dq_scaled7,Dq_scaled8,Dq_scaled9]
    return q_scaled_dot