# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:23:25 2021

@author: bayat2
"""
import openmdao.api as om
import dymos as dm
from WT_Continuous_class import WT_Continuous

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(ode_class=WT_Continuous,
                 transcription=dm.Radau(num_segments=5, order=4))
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
                fix_initial=True,
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

                     