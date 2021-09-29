# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:45:02 2021

@author: bayat2
"""
import numpy as np
from Inner_simulation_func import Inner_Simulation
from obj_inner_func import obj_inner
from obj_inner_test_func import obj_inner_test

def Innerloop_CMAES(x_scaled,i_speed,Outer_loop_params):
    
    x_1_b=Outer_loop_params.x_1_b;  x_1_a=Outer_loop_params.x_1_a;
    x_2_b=Outer_loop_params.x_2_b;  x_2_a=Outer_loop_params.x_2_a;
    x_3_b=Outer_loop_params.x_3_b;  x_3_a=Outer_loop_params.x_3_a;
    x_4_b=Outer_loop_params.x_4_b;  x_4_a=Outer_loop_params.x_4_a;

    x1=(x_1_b+x_1_a)/2+(x_1_b-x_1_a)/2*(x_scaled[0]);
    x2=(x_2_b+x_2_a)/2+(x_2_b-x_2_a)/2*(x_scaled[1]);
    x3=(x_3_b+x_3_a)/2+(x_3_b-x_3_a)/2*(x_scaled[2]);
    x4=(x_4_b+x_4_a)/2+(x_4_b-x_4_a)/2*(x_scaled[3]);
    
    x=np.array([[x1],[x2],[x3],[x4]])
    
    wind_speed=Outer_loop_params.wind_speed;
    wind_Probablity=Outer_loop_params.wind_Probablity;
    
    Obj_innerloop=0;
    
    wind_type=1                    #wind type
    wind_avg=wind_speed[i_speed]
    obj_inner_val=obj_inner(x,wind_avg,wind_type)
    #obj_inner_val=obj_inner_test(x,wind_avg,wind_type)
    
    Obj_innerloop=8760*1e-9*wind_Probablity[i_speed]*obj_inner_val+Obj_innerloop; #GWh
    #return Obj_innerloop[0]
    #return obj_inner_val
    return Obj_innerloop[0]