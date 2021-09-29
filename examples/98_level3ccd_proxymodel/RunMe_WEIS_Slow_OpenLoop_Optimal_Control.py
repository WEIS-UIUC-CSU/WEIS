# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:03:41 2021

@author: bayat2

"""

#export LD_LIBRARY_PATH=/home/bayat2/ipopt/lib:$LD_LIBRARY_PATH
import os
import random, string
import numpy as np
from auxdata_gen_class import auxdata
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
#%% input data
"""--------------------------Input data-----------------------"""
x=np.array([[0.0254], #tower's base thickness [m]
            [0.0114], #tower's tip thickness [m]
            [4.1251], #tower's tip external diamter [m]
            [95.57]]) #tower's length [m]
wind_avg=12     #average wind speed
wind_type=1    #wind type
auxdata_instance=auxdata(x,wind_avg,wind_type)

t0 = 0     # [s] 
tf=100*1   # finale simulation time [s]

# i f we have no constraint, and pitch mechanism is inactive, then oprimal generator
# torque signal signal can be obtained analyticallt and can be written as:
# {u_{gen}_{ref}=k_star*\omega_{star}^2. 
# reference: Pao, Lucy Y., and Kathryn E. Johnson. "A tutorial on the dynamics and control of wind turbines and wind farms." 2009 American Control Conference. IEEE, 2009.
# so here for intial guess we used this control law:

#%%
"""---------------Initial guess of geenrator torque------------------"""
lambda_star=7.6                 #tip speed ratio that yields maximum Cp
cp_star=0.4978                  #max Cp
t_guess=np.linspace(0,tf,360)
omega_star=lambda_star*auxdata_instance.v(t_guess)/auxdata_instance.R_rotor #optimal Omega
k_star=np.pi*auxdata_instance.rho_air*auxdata_instance.R_rotor**5*cp_star/(2*lambda_star**3);
gen_torque_star=k_star*omega_star**2    #opmtimal generator torque

#scaling guess
#u_scaled=2*(u-u_{min})/(u_{max}-u_{min})-1
omega_star_scaled=2*(omega_star-0)/(auxdata_instance.A_scaled[3,0]*2-0)-1
gen_torque_star_scaled=2*(gen_torque_star-0)/(auxdata_instance.A_scaled[-1,0]*2-0)-1

#%%
"""--------------------------Dynamics-----------------------"""
class WT_Continuous(om.ExplicitComponent):
    
#    def __init__(self,*args, **kwargs):
#        self.progress_prints = False
#        super().__init__(*args, **kwargs)
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    #    self.options.declare('gg', gg)

    def setup(self):
        nn = self.options['num_nodes']
        #print(self.options['gg'])

        """Inputs (time, states, controls) &
           Outputs(state rates, path const, and objective rate) """
        
        #%%time
        self.add_input('t',
                       shape=(nn,),
                       desc='value of current time',
                       units='s')
        
        #%%states
        self.add_input('q1_scaled',
                       shape=(nn,),
                       desc='v_x (platform surge rate)',
                       units='m/s')
        
        self.add_input('q2_scaled',
                       shape=(nn,),
                       desc='v_z (platfrom heave rate)',
                       units='m/s')
        
        self.add_input('q3_scaled',
                       shape=(nn,),
                       desc='\omega_y (Platfrom pitch rate)',
                       units='rad/s')
        
        self.add_input('q4_scaled',
                       shape=(nn,),
                       desc='\Omega (Rotor angular velocity)',
                       units='rad/s')
        
        self.add_input('q5_scaled',
                       shape=(nn,),
                       desc='XX (Pltform surge)',
                       units='m')
        
        self.add_input('q6_scaled',
                       shape=(nn,),
                       desc='ZZ (Platform heave)',
                       units='m')
        
        self.add_input('q7_scaled',
                       shape=(nn,),
                       desc='\theta (platfrom pitch)',
                       units='rad')
        
        self.add_input('q8_scaled',
                       shape=(nn,),
                       desc='\beta (blade pitch)',
                       units='rad')
        
        self.add_input('q9_scaled',
                       shape=(nn,),
                       desc='T_{gen} (generator torque)',
                       units='N*m')
        
        #%%controls
        self.add_input('u1_scaled',
                       shape=(nn,),
                       desc='Tgen_dot_scaled',
                       units='N*m/s')
        
        self.add_input('u2_scaled',
                       shape=(nn,),
                       desc='beta_dot_scaled',
                       units='rad/s')
        
        #%%outputs
        self.add_output('q1_dot_scaled',
                        shape=(nn,),
                        desc='v_x_dot',
                        units='m/(s**2)')
        
        self.add_output('q2_dot_scaled',
                        shape=(nn,),
                        desc='v_y_dot',
                        units='m/(s**2)')
        
        self.add_output('q3_dot_scaled',
                        shape=(nn,),
                        desc='\omega_y_dot',
                        units='rad/(s**2)')
        
        self.add_output('q4_dot_scaled',
                        shape=(nn,),
                        desc='\Omega_dot',
                        units='rad/(s**2)')
        
        self.add_output('q5_dot_scaled',
                        shape=(nn,),
                        desc='XX_dot',
                        units='m/s')
        
        self.add_output('q6_dot_scaled',
                        shape=(nn,),
                        desc='ZZ_dot',
                        units='m/s')
        
        self.add_output('q7_dot_scaled',
                        shape=(nn,),
                        desc='\theta_dot',
                        units='rad/s')
        
        self.add_output('q8_dot_scaled',
                        shape=(nn,),
                        desc='\beta_dot',
                        units='rad/s')
        
        self.add_output('q9_dot_scaled',
                        shape=(nn,),
                        desc='T_{gen}_dot',
                        units='N*m/s')
        
        self.add_output('obj_dot',
                        val=np.zeros(nn),
                        desc='objective dot',
                        units='N*m/s')
        
        #%% path output
        self.add_output('tansion_at_tower_root_const',
                        val=np.zeros(nn),
                        desc='tansion_at_tower_root_const',
                        units='N/(m**2)')
        
        self.add_output('Max_power_const',
                        val=np.zeros(nn),
                        desc='Max_power_const',
                        units='N*d')


        #%% Use OpenMDAO's ability to automatically determine a sparse "coloring" of the jacobian
        # for this ODE component.
        #self.declare_coloring(wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='fd')
        #self.declare_partials(of='*', wrt='*', method='fd')
        
    #%% define dynamics
    def compute(self, inputs, outputs):
        t = inputs['t']
        q1_scaled = inputs['q1_scaled']
        q2_scaled = inputs['q2_scaled']
        q3_scaled = inputs['q3_scaled']
        q4_scaled = inputs['q4_scaled']
        q5_scaled = inputs['q5_scaled']
        q6_scaled = inputs['q6_scaled']
        q7_scaled = inputs['q7_scaled']
        q8_scaled = inputs['q8_scaled']
        q9_scaled = inputs['q9_scaled']
        
        u1_scaled = inputs['u1_scaled']
        u2_scaled = inputs['u2_scaled']
        
        q_scaled=np.c_[q1_scaled, q2_scaled, q3_scaled, q4_scaled, q5_scaled, q6_scaled, q7_scaled, q8_scaled, q9_scaled ]
        Control_scaled=np.c_[u1_scaled, u2_scaled]
        
        etha=auxdata_instance.etha;
        
        #scaled data
        A_scaled=auxdata_instance.A_scaled
        B_scaled=auxdata_instance.B_scaled
        C_scaled=auxdata_instance.C_scaled
        D_scaled=auxdata_instance.D_scaled

        qq=q_scaled;
        for i in np.arange(q_scaled.shape[1]):
            qq[:,i]=A_scaled[i]+B_scaled[i,i]*q_scaled[:,i]
            
        U1=C_scaled[0]+D_scaled[0,0]*u1_scaled
        U2=C_scaled[1]+D_scaled[1,1]*u2_scaled  
        
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
        
        lambda_R=R_rotor*qq[:,3]/(U_rel)
        Cp_vec=Cp.ev(lambda_R,qq[:,7])
        Ct_vec=Ct.ev(lambda_R,qq[:,7])
        #Cp_vec=np.linspace(0,0.45,len(t))
        #Ct_vec=np.linspace(0,0.9,len(t))
        
        tau_a=1/2*rho_air*np.pi*R_rotor**3*Cp_vec/lambda_R*U_rel**2 #aerodynamic torque
        tau_g=qq[:,8] #generator torque
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
        
        dq8=U2
        dq9=U1
        
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
        
        #%%
        """---------------External Force and Moment--------------"""
        
        outputs['q1_dot_scaled']=Dq_scaled1
        outputs['q2_dot_scaled']=Dq_scaled2
        outputs['q3_dot_scaled']=Dq_scaled3
        outputs['q4_dot_scaled']=Dq_scaled4
        outputs['q5_dot_scaled']=Dq_scaled5
        outputs['q6_dot_scaled']=Dq_scaled6
        outputs['q7_dot_scaled']=Dq_scaled7
        outputs['q8_dot_scaled']=Dq_scaled8
        outputs['q9_dot_scaled']=Dq_scaled9
        
        outputs['obj_dot']=objective
        
        outputs['tansion_at_tower_root_const']=path_con_1
        outputs['Max_power_const']=path_con_2


"""---------Defines states, controls, and outputs------"""
# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(ode_class=WT_Continuous,
                 transcription=dm.Radau(num_segments=10, order=8))
#dm.GaussLobatto(num_segments=num_segments,
#                            order=transcription_order,
#                            compressed=compressed)

traj.add_phase(name='phase0', phase=phase)
        

# Set the time options
phase.set_time_options(units='s',
                       fix_initial=True,
                       fix_duration=True,
                       initial_val=0.0,
                       duration_val=100.0,
                       targets='t')

# Set the state options
phase.add_state(name='q1_scaled', 
                units='m/s',
                rate_source='q1_dot_scaled',
                targets='q1_scaled',
                fix_initial=True,
                lower=-1,
                upper=1,)    
                       
phase.add_state(name='q2_scaled', 
                units='m/s',
                rate_source='q2_dot_scaled',
                targets='q2_scaled',
                fix_initial=True,
                lower=-1,
                upper=1,)  

phase.add_state(name='q3_scaled', 
                units='rad/s',
                rate_source='q3_dot_scaled',
                targets='q3_scaled',
                fix_initial=True,
                lower=-1,
                upper=1,)  

phase.add_state(name='q4_scaled', 
                units='rad/s',
                rate_source='q4_dot_scaled',
                targets='q4_scaled',
                fix_initial=False,
                lower=-1,
                upper=1,)  

phase.add_state(name='q5_scaled', 
                units='m',
                rate_source='q5_dot_scaled',
                targets='q5_scaled',
                fix_initial=True,
                lower=-1,
                upper=1,)  

phase.add_state(name='q6_scaled', 
                units='m',
                rate_source='q6_dot_scaled',
                targets='q6_scaled',
                fix_initial=True,
                lower=-1,
                upper=1,)  

phase.add_state(name='q7_scaled', 
                units='rad',
                rate_source='q7_dot_scaled',
                targets='q7_scaled',
                fix_initial=True,
                lower=-1,
                upper=1,)  

phase.add_state(name='q8_scaled', 
                units='rad',
                rate_source='q8_dot_scaled',
                targets='q8_scaled',
                lower=-1,
                upper=1,)  

phase.add_state(name='q9_scaled', 
                units='N*m',
                rate_source='q9_dot_scaled',
                targets='q9_scaled',
                lower=-1,
                upper=1,)  

phase.add_state(name='obj', 
                units='N*m',
                rate_source='obj_dot',
                lower=-0.5,
                upper=0,
                fix_initial=True)

# define the control
phase.add_control(name='u1_scaled', 
                  units='N*m/s',
                  desc='Tgen_dot_scaled',
                  targets='u1_scaled',
                  lower=-1, 
                  upper=1, 
                  continuity=True,
                  rate_continuity=True, 
                  )  

phase.add_control(name='u2_scaled', 
                  units='rad/s',
                  desc='beta_dot_scaled',
                  targets='u2_scaled',
                  lower=-1, 
                  upper=1, 
                  continuity=True,
                  rate_continuity=True, 
                  )  

#define objective to minimize
phase.add_objective(name='obj', loc='final')

phase.add_boundary_constraint(name='q4_scaled', 
                              loc='initial', lower=2*1.2671/auxdata_instance.omega_max-1, upper=2*1.2671/auxdata_instance.omega_max-1+1e-3)

phase.add_path_constraint(name='tansion_at_tower_root_const',
                          constraint_name='tansion_at_tower_root_const',
                          lower=-90,
                          upper=90)

phase.add_path_constraint(name='Max_power_const',
                          constraint_name='Max_power_const',
                          lower=0,
                          upper=1)


#In addition to these default values, any output of the ODE can be added to the timeseries output 
phase.add_timeseries_output(name='tansion_at_tower_root_const',output_name='tansion_at_tower_root_const')
phase.add_timeseries_output(name='Max_power_const',output_name='Max_power_const')
#(sim_case.get_val('traj.phase0.timeseries.Path_const_1')).flatten()

""" Configuration """
# Set the driver.
#options={'maxiter': 300}
#p.driver = om.ScipyOptimizeDriver()
#p.driver.opt_settings=options

_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'IPOPT'
p.driver.opt_settings['print_level'] = 5
p.driver.options['print_results'] = False
#p.driver.opt_settings['mu_init'] = 1e-3
p.driver.opt_settings['max_iter'] = 500
#p.driver.opt_settings['acceptable_tol'] = 1e-3
#p.driver.opt_settings['constr_viol_tol'] = 1e-3
#p.driver.opt_settings['compl_inf_tol'] = 1e-3
#p.driver.opt_settings['acceptable_iter'] = 0
#p.driver.opt_settings['tol'] = 1e-3
#p.driver.opt_settings['nlp_scaling_method'] = 'none'
#p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence


# Allow OpenMDAO to automatically determine our sparsity pattern
# for TOTAL derivatives. Doing so can significantly speed up the
# execution of dymos.
p.driver.declare_coloring()

# Setup the problem
p.setup()

# Now that the OpenMDAO problem is setup, we can guess the
# values of time, states, and controls.

# States and controls here use a linearly interpolated
# initial guess along the trajectory.
#p['traj.phase0.states:x0'] = phase.interpolate(ys=[1, 0], nodes='state_input')
p.set_val('traj.phase0.states:q1_scaled',
          phase.interpolate(ys=[0, 0], nodes='state_input'),
          units='m/s')

p.set_val('traj.phase0.states:q2_scaled',
          phase.interpolate(ys=[0, 0], nodes='state_input'),
          units='m/s')

p.set_val('traj.phase0.states:q3_scaled',
          phase.interpolate(ys=[0, 0], nodes='state_input'),
          units='rad/s')

p.set_val('traj.phase0.states:q4_scaled',
          phase.interpolate(xs=t_guess,ys=omega_star_scaled, nodes='state_input'),
          units='rad/s')

p.set_val('traj.phase0.states:q5_scaled',
          phase.interpolate(ys=[0, 0], nodes='state_input'),
          units='m')

p.set_val('traj.phase0.states:q6_scaled',
          phase.interpolate(ys=[0, 0], nodes='state_input'),
          units='m')

p.set_val('traj.phase0.states:q7_scaled',
          phase.interpolate(ys=[0, 0], nodes='state_input'),
          units='rad')

p.set_val('traj.phase0.states:q8_scaled',
          phase.interpolate(xs=t_guess,ys=(1/11*(wind_avg)-14/11)*np.ones(len(t_guess)), nodes='state_input'),
          units='rad')

p.set_val('traj.phase0.states:q9_scaled',
          phase.interpolate(xs=t_guess,ys=gen_torque_star_scaled, nodes='state_input'),
          units='N*m')

p.set_val('traj.phase0.states:obj',
          phase.interpolate(ys=[0, -0.5], nodes='state_input'),
          units='N*m')

p.set_val('traj.phase0.controls:u1_scaled',
          phase.interpolate(ys=[0, 0], nodes='control_input'),
          units='N*m/s')

p.set_val('traj.phase0.controls:u2_scaled',
          phase.interpolate(ys=[0, 0], nodes='control_input'),
          units='rad/s')


# Use Dymos' run_problem method to run the driver, simulate the results,
# and record the results to 'dymos_solution.db' and 'dymos_simulation.db'.

#p.model.traj.phases.phase0.set_refine_options(refine=True)
#dm.run_problem(p, refine_iteration_limit=10)

sol_name = ''.join(random.choices(string.ascii_letters, k=16))
sim_name= ''.join(random.choices(string.ascii_letters, k=16))

sol_name=sol_name+'.db'
sim_name=sim_name+'.db'

#sol_name='dymos_solution.db'
#sim_name='dymos_simulation.db'

#dm.run_problem(p, simulate=True)
dm.run_problem(problem=p,simulate=True,solution_record_file=sol_name,simulation_record_file=sim_name)

# Load the solution and simulation files.
sol_case = om.CaseReader(sol_name).get_case('final')
sim_case = om.CaseReader(sim_name).get_case('final')


time_vec_sim=(sim_case.get_val('traj.phase0.timeseries.time')).flatten()
time_vec_sol=(sol_case.get_val('traj.phase0.timeseries.time')).flatten()

q1_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.states:q1_scaled')).flatten()
q1_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.states:q1_scaled')).flatten()

q2_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.states:q2_scaled')).flatten()
q2_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.states:q2_scaled')).flatten()

q3_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.states:q3_scaled')).flatten()
q3_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.states:q3_scaled')).flatten()

q4_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.states:q4_scaled')).flatten()
q4_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.states:q4_scaled')).flatten()

q5_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.states:q5_scaled')).flatten()
q5_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.states:q5_scaled')).flatten()

q6_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.states:q6_scaled')).flatten()
q6_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.states:q6_scaled')).flatten()

q7_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.states:q7_scaled')).flatten()
q7_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.states:q7_scaled')).flatten()

q8_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.states:q8_scaled')).flatten()
q8_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.states:q8_scaled')).flatten()

q9_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.states:q9_scaled')).flatten()
q9_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.states:q9_scaled')).flatten()

obj_vec_sim=(sim_case.get_val('traj.phase0.timeseries.states:obj')).flatten()
obj_vec_sol=(sol_case.get_val('traj.phase0.timeseries.states:obj')).flatten()

u1_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.controls:u1_scaled')).flatten()
u1_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.controls:u1_scaled')).flatten()

u2_vec_scaled_sim=(sim_case.get_val('traj.phase0.timeseries.controls:u2_scaled')).flatten()
u2_vec_scaled_sol=(sol_case.get_val('traj.phase0.timeseries.controls:u2_scaled')).flatten()

tansion_at_tower_root_const_vec_sim=(sim_case.get_val('traj.phase0.timeseries.tansion_at_tower_root_const')).flatten()
tansion_at_tower_root_const_vec_sol=(sol_case.get_val('traj.phase0.timeseries.tansion_at_tower_root_const')).flatten()

Max_power_const_vec_sim=(sim_case.get_val('traj.phase0.timeseries.Max_power_const')).flatten()
Max_power_const_vec_sol=(sol_case.get_val('traj.phase0.timeseries.Max_power_const')).flatten()

q_vec_scaled_sim=np.c_[np.reshape(q1_vec_scaled_sim,(-1,1)),\
                np.reshape(q2_vec_scaled_sim,(-1,1)),\
                np.reshape(q3_vec_scaled_sim,(-1,1)),\
                np.reshape(q4_vec_scaled_sim,(-1,1)),\
                np.reshape(q5_vec_scaled_sim,(-1,1)),\
                np.reshape(q6_vec_scaled_sim,(-1,1)),\
                np.reshape(q7_vec_scaled_sim,(-1,1)),\
                np.reshape(q8_vec_scaled_sim,(-1,1)),\
                np.reshape(q9_vec_scaled_sim,(-1,1))]
    
q_vec_scaled_sol=np.c_[np.reshape(q1_vec_scaled_sol,(-1,1)),\
                np.reshape(q2_vec_scaled_sol,(-1,1)),\
                np.reshape(q3_vec_scaled_sol,(-1,1)),\
                np.reshape(q4_vec_scaled_sol,(-1,1)),\
                np.reshape(q5_vec_scaled_sol,(-1,1)),\
                np.reshape(q6_vec_scaled_sol,(-1,1)),\
                np.reshape(q7_vec_scaled_sol,(-1,1)),\
                np.reshape(q8_vec_scaled_sol,(-1,1)),\
                np.reshape(q9_vec_scaled_sol,(-1,1))]    

q_vec_sim=q_vec_scaled_sim
q_vec_sol=q_vec_scaled_sol

for i in np.arange(np.shape(q_vec_scaled_sim)[1]):
    q_vec_sim[:,i]=auxdata_instance.A_scaled[i]+auxdata_instance.B_scaled[i,i]*q_vec_scaled_sim[:,i]
    q_vec_sol[:,i]=auxdata_instance.A_scaled[i]+auxdata_instance.B_scaled[i,i]*q_vec_scaled_sol[:,i]

U1_vec_sim=auxdata_instance.C_scaled[0]+auxdata_instance.D_scaled[0,0]*u1_vec_scaled_sim
U1_vec_sol=auxdata_instance.C_scaled[0]+auxdata_instance.D_scaled[0,0]*u1_vec_scaled_sol

U2_vec_sim=auxdata_instance.C_scaled[1]+auxdata_instance.D_scaled[1,1]*u2_vec_scaled_sim
U2_vec_sol=auxdata_instance.C_scaled[1]+auxdata_instance.D_scaled[1,1]*u2_vec_scaled_sol

theta_sim=q_vec_sim[:,6]
theta_sol=q_vec_sol[:,6]

R11_sim=np.cos(theta_sim)
R11_sol=np.cos(theta_sol)
R12_sim=np.sin(theta_sim)
R12_sol=np.sin(theta_sol)
R21_sim=-np.sin(theta_sim)
R21_sol=-np.sin(theta_sol)
R22_sim=np.cos(theta_sim)
R22_sol=np.cos(theta_sol)

v_wind_sim=np.r_[np.reshape(auxdata_instance.v(time_vec_sim),(1,-1)),0*np.reshape(auxdata_instance.v(time_vec_sim),(1,-1))]
v_wind_sol=np.r_[np.reshape(auxdata_instance.v(time_vec_sol),(1,-1)),0*np.reshape(auxdata_instance.v(time_vec_sol),(1,-1))]

v_wind_b_1_sim=R11_sim*v_wind_sim[0,:]+R21_sim*v_wind_sim[1,:]
v_wind_b_2_sim=R12_sim*v_wind_sim[0,:]+R22_sim*v_wind_sim[1,:]

v_wind_b_1_sol=R11_sol*v_wind_sol[0,:]+R21_sol*v_wind_sol[1,:]
v_wind_b_2_sol=R12_sol*v_wind_sol[0,:]+R22_sol*v_wind_sol[1,:]

v_wind_b_sim=np.r_[np.reshape(v_wind_b_1_sim,(1,-1)),np.reshape(v_wind_b_2_sim,(1,-1))]
v_wind_b_sol=np.r_[np.reshape(v_wind_b_1_sol,(1,-1)),np.reshape(v_wind_b_2_sol,(1,-1))]

v_r_1_sim=q_vec_sim[:,0]+auxdata_instance.Dr*q_vec_sim[:,2]
v_r_1_sol=q_vec_sol[:,0]+auxdata_instance.Dr*q_vec_sol[:,2]

U_rel_sim=v_wind_b_sim[0,:]-v_r_1_sim
U_rel_sol=v_wind_b_sol[0,:]-v_r_1_sol

lambda_R_sim=auxdata_instance.R_rotor*q_vec_sim[:,3]/U_rel_sim
lambda_R_sol=auxdata_instance.R_rotor*q_vec_sol[:,3]/U_rel_sol

Cp_vec_sim=auxdata_instance.Cpfunc.ev(lambda_R_sim,q_vec_sim[:,7])
Cp_vec_sol=auxdata_instance.Cpfunc.ev(lambda_R_sol,q_vec_sol[:,7])

Ct_vec_sim=auxdata_instance.Ctfunc.ev(lambda_R_sim,q_vec_sim[:,7])
Ct_vec_sol=auxdata_instance.Ctfunc.ev(lambda_R_sol,q_vec_sol[:,7])

P_a_sim=Cp_vec_sim*0.5*auxdata_instance.rho_air*np.pi*auxdata_instance.R_rotor**2*(U_rel_sim)**3*auxdata_instance.etha
P_a_sol=Cp_vec_sol*0.5*auxdata_instance.rho_air*np.pi*auxdata_instance.R_rotor**2*(U_rel_sol)**3*auxdata_instance.etha

P_u_sim=q_vec_sim[:,3]*q_vec_sim[:,8]*auxdata_instance.etha
P_u_sol=q_vec_sol[:,3]*q_vec_sol[:,8]*auxdata_instance.etha

taw_a_sim=1/2*auxdata_instance.rho_air*np.pi*auxdata_instance.R_rotor**3*Cp_vec_sim/lambda_R_sim*U_rel_sim**2
taw_a_sol=1/2*auxdata_instance.rho_air*np.pi*auxdata_instance.R_rotor**3*Cp_vec_sol/lambda_R_sol*U_rel_sol**2

taw_g_sim=q_vec_sim[:,8]
taw_g_sol=q_vec_sol[:,8]

omega_dot_sim=1/auxdata_instance.J_rotor*(taw_a_sim-taw_g_sim)
omega_dot_sol=1/auxdata_instance.J_rotor*(taw_a_sol-taw_g_sol)

P_inter_sim=auxdata_instance.J_rotor*(q_vec_sim[:,3])*omega_dot_sim*auxdata_instance.etha
P_inter_sol=auxdata_instance.J_rotor*(q_vec_sol[:,3])*omega_dot_sol*auxdata_instance.etha


#%% 
"""-----------------------Plot Results-----------------"""
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,q_vec_sim[:,0],'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,q_vec_sol[:,0],'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$v_x \, \mathrm{[m/s]\,\,(Surge\,\, rate)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,q_vec_sim[:,1],'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,q_vec_sol[:,1],'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$v_z \, \mathrm{[m/s]\,\,(Heave\,\, rate)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,q_vec_sim[:,2],'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,q_vec_sol[:,2],'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$\omega_p \, \mathrm{[rad/s]\,\,(Platform\,\, pitch\,\, rate)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,q_vec_sim[:,3],'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,q_vec_sol[:,3],'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$\omega_r \, \mathrm{[rad/s]\,\,(rotor\,\, ratational\,\, speed)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,q_vec_sim[:,4],'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,q_vec_sol[:,4],'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$x\, \mathrm{[m]\,\,(Platform\,\, surge)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,q_vec_sim[:,5],'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,q_vec_sol[:,5],'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$z\, \mathrm{[m]\,\,(Platform\,\, heave)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,q_vec_sim[:,6],'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,q_vec_sol[:,6],'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel(r'$\theta_p\, \mathrm{[rad]\,\,(Platform\,\, pitch)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,q_vec_sim[:,7],'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,q_vec_sol[:,7],'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel(r'$\theta_b\, \mathrm{[rad]\,\,(Blade\,\, pitch)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,1e-6*q_vec_sim[:,8],'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,1e-6*q_vec_sol[:,8],'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$T_{gen}\, \mathrm{[MNm]\,\,(Generator\,\, Torque)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,1e-3*U1_vec_sim,'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,1e-3*U1_vec_sol,'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$\dot{T}_{gen}\, \mathrm{[kNm/s]\,\,(control\,\, signal)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,U2_vec_sim,'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,U2_vec_sol,'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel(r'$\dot{\theta}_{b}\, \mathrm{[rad/s]\,\,(control\,\, signal)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,Cp_vec_sim,'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,Cp_vec_sol,'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$C_p\, \mathrm{[-]\,\,(Power\,\, coefficient)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,Ct_vec_sim,'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,Ct_vec_sol,'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$C_t\, \mathrm{[-]\,\,(Thrust\,\, coefficient)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,tansion_at_tower_root_const_vec_sim,'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,tansion_at_tower_root_const_vec_sol,'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$\sigma\, \mathrm{[MPa]\,\,(Tower\,\, maximum\,\,stress)}$',fontsize=15)
ax.legend(prop = fontP)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_vec_sim,Max_power_const_vec_sim,'-', linewidth=2,label='sim')
ax.plot(time_vec_sol,Max_power_const_vec_sol,'o', markersize=8,label='sol',markerfacecolor='none',markeredgewidth=2)
ax.set_xlabel('$t\, \mathrm{[s]}$',fontsize=15)
ax.set_ylabel('$P_{gen}/{5MW}\,\, \mathrm{[-]\,\,(Genrator\,\, power)}$',fontsize=15)
ax.legend(prop = fontP)

os.remove(sol_name)
os.remove(sim_name)