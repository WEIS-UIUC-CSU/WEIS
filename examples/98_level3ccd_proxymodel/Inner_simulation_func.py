# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:51:07 2021

@author: bayat2
"""
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
from obj_inner_func import obj_inner

def Inner_Simulation(x_scaled,Outer_loop_params,i_speed,wind_speed_scalar):
    
    x_1_b=Outer_loop_params.x_1_b;  x_1_a=Outer_loop_params.x_1_a;
    x_2_b=Outer_loop_params.x_2_b;  x_2_a=Outer_loop_params.x_2_a;
    x_3_b=Outer_loop_params.x_3_b;  x_3_a=Outer_loop_params.x_3_a;
    x_4_b=Outer_loop_params.x_4_b;  x_4_a=Outer_loop_params.x_4_a;

    #unscaling
    x1=(x_1_b+x_1_a)/2+(x_1_b-x_1_a)/2*(x_scaled[0]);
    x2=(x_2_b+x_2_a)/2+(x_2_b-x_2_a)/2*(x_scaled[1]);
    x3=(x_3_b+x_3_a)/2+(x_3_b-x_3_a)/2*(x_scaled[2]);
    x4=(x_4_b+x_4_a)/2+(x_4_b-x_4_a)/2*(x_scaled[3]);
    
    x=np.array([[x1],[x2],[x3],[x4]])
    
    
    wind_avg=wind_speed_scalar     #average wind speed
    wind_type=1                    #wind type
    Obj_innerloop=obj_inner(x,wind_avg,wind_type)
    return Obj_innerloop