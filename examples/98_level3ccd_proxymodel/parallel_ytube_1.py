#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 18:22:24 2021

@author: bayat2
"""
import multiprocessing
import numpy as np

lambda_pop=10
wind_index=np.tile(np.array([[0],[1],[2],[3],[4],[5]]),(lambda_pop,1))
arx_added=np.random.rand(4,60)

def obj(x0,x1,x2,x3,y):
    z=x0+x1+x2+x3+y
    print(z)
    return z

processes=[]

for indx in range(60):
    p=multiprocessing.Process(target=obj,args=[arx_added[0,indx],arx_added[1,indx],arx_added[2,indx],arx_added[3,indx],wind_index[indx]])
    p.start()
    processes.append(p)
    
for process in processes:
    process.join()    