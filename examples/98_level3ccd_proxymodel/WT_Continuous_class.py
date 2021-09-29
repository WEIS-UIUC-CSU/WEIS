# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:51:24 2021

@author: bayat2
"""

import numpy as np
import openmdao.api as om

"""
from auxdata_gen_class import auxdata

jj_wind=6
x=np.array([[0.0254], #tower's base thickness [m]
            [0.0114], #tower's tip thickness [m]
            [4.1251], #tower's tip external diamter [m]
            [95.57]]) #tower's length [m]
wind_avg=6     #average wind speed
wind_type=1    #wind type
auxdata_instance=auxdata(x,wind_avg,wind_type)"""

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
        
        #time
        self.add_input('t',
                       shape=(nn,),
                       desc='value of current time',
                       units='s')
        
        #states
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
        
        #controls
        self.add_input('u1_scaled',
                       shape=(nn,),
                       desc='Tgen_dot_scaled',
                       units='N*m/s')
        
        self.add_input('u2_scaled',
                       shape=(nn,),
                       desc='beta_dot_scaled',
                       units='rad/s')
        
        #outputs
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
        
        self.add_output('tansion_at_tower_root_const',
                        val=np.zeros(nn),
                        desc='tansion_at_tower_root_const',
                        units='N/(m**2)')
        
        self.add_output('Max_power_const',
                        val=np.zeros(nn),
                        desc='Max_power_const',
                        units='N*d')


        # Use OpenMDAO's ability to automatically determine a sparse "coloring" of the jacobian
        # for this ODE component.
        #self.declare_coloring(wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='fd')
        #self.declare_partials(of='*', wrt='*', method='fd')

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
        
        outputs['q1_dot_scaled']=q1_scaled
        outputs['q2_dot_scaled']=q2_scaled
        outputs['q3_dot_scaled']=q3_scaled
        outputs['q4_dot_scaled']=q4_scaled
        outputs['q5_dot_scaled']=q5_scaled
        outputs['q6_dot_scaled']=q6_scaled
        outputs['q7_dot_scaled']=q7_scaled
        outputs['q8_dot_scaled']=q8_scaled
        outputs['q9_dot_scaled']=q9_scaled
        
        outputs['obj_dot']=q9_scaled
        
        outputs['tansion_at_tower_root_const']=q7_scaled
        outputs['Max_power_const']=q8_scaled
 
