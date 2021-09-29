# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:07:52 2021

@author: bayat2
"""
import numpy as np
from CMAES_func import CMAES_Func
from InnerLoop_CMAES_func import Innerloop_CMAES
import dill

wind_speed = np.array([6,8,10,12,14,16])
wind_Probablity =np.array([0.15,0.2,0.3,0.2,0.1,0.05])

class Outer_loop_params_class:
    def __init__(self):
        self.wind_speed=wind_speed
        self.wind_Probablity=wind_Probablity
        self.x_n=np.array([0.027, 0.019, 3.87, 77.6])
        self.x_1_b=1.6*self.x_n[0]
        self.x_1_a=0.8037*self.x_n[0]
        self.x_2_b=1.5*self.x_n[1]
        self.x_2_a=0.6*self.x_n[1]
        self.x_3_b=6.5
        self.x_3_a=0.83*self.x_n[2]
        self.x_4_b=1.4*self.x_n[3]
        self.x_4_a=0.8*self.x_n[3]

OLP_instance=Outer_loop_params_class()
        
#x0_unscaled=np.array([0.0218, 0.0115, 3.63, 88.6])
x0_unscaled=np.array([0.027, 0.019, 3.87, 77.6])
x0_1=2*(x0_unscaled[0]-OLP_instance.x_1_a)/(OLP_instance.x_1_b-OLP_instance.x_1_a)-1
x0_2=2*(x0_unscaled[1]-OLP_instance.x_2_a)/(OLP_instance.x_2_b-OLP_instance.x_2_a)-1
x0_3=2*(x0_unscaled[2]-OLP_instance.x_3_a)/(OLP_instance.x_3_b-OLP_instance.x_3_a)-1
x0_4=2*(x0_unscaled[3]-OLP_instance.x_4_a)/(OLP_instance.x_4_b-OLP_instance.x_4_a)-1

x0=np.array([[x0_1], [x0_2], [x0_3], [x0_4]])

N=4
funcval_stop_outerloop=40*50
#funcval_stop_outerloop=2
lambda_pop=50
#lambda_pop=2

#obj_func=lambda x_scaled,i_speed: Innerloop_CMAES(x_scaled,i_speed,OLP_instance)
obj_func=[]
[x_scaled_opt,fmin,x_opt_vector,f_opt_vector,x_total_vec,f_total_vec]=CMAES_Func(obj_func,N,funcval_stop_outerloop,lambda_pop,x0,OLP_instance)

x_1_b=OLP_instance.x_1_b;  x_1_a=OLP_instance.x_1_a;
x_2_b=OLP_instance.x_2_b;  x_2_a=OLP_instance.x_2_a;
x_3_b=OLP_instance.x_3_b;  x_3_a=OLP_instance.x_3_a;
x_4_b=OLP_instance.x_4_b;  x_4_a=OLP_instance.x_4_a;

x_unscaled_opt=np.zeros(4)

x_unscaled_opt[0]=(x_1_b+x_1_a)/2+(x_1_b-x_1_a)/2*(x_scaled_opt[0])
x_unscaled_opt[1]=(x_2_b+x_2_a)/2+(x_2_b-x_2_a)/2*(x_scaled_opt[1])
x_unscaled_opt[2]=(x_3_b+x_3_a)/2+(x_3_b-x_3_a)/2*(x_scaled_opt[2])
x_unscaled_opt[3]=(x_4_b+x_4_a)/2+(x_4_b-x_4_a)/2*(x_scaled_opt[3])

#filename ='CCD_OL_Results_N4_WW_Ctr90_lambda50_fval_2000.pkl'
#dill.dump_session(filename)

#dill.load_session(filename)