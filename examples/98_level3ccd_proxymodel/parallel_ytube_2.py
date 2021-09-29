#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 19:57:15 2021

@author: bayat2
"""

import concurrent.futures
import numpy as np
import time
from parallel_ytube_2_obj import obj

lambda_pop=10
wind_index=np.tile(np.array([[0],[1],[2],[3],[4],[5]]),(lambda_pop,1))
arx_added=np.random.rand(4,60)

"""
def obj(x0,x1,x2,x3,y):
    z=x0+x1+x2+x3+y
    print(z)
    return z"""

"""
def obj(x,y):
    z=x[0]+x[1]+x[2]+x[3]+y
    if x[0]==arx_added[0,0]:
        time.sleep(1)
    return z"""

"""
indx=0
with concurrent.futures.ProcessPoolExecutor() as executor:
    #p=executor.submit(obj,arx_added[0,indx],arx_added[1,indx],arx_added[2,indx],arx_added[3,indx],wind_index[indx])
    p=executor.submit(obj,[arx_added[0,indx],arx_added[1,indx],arx_added[2,indx],arx_added[3,indx]],wind_index[indx])
    print(p.result())"""

processes=[]
with concurrent.futures.ProcessPoolExecutor() as executor:
    for indx in range(60):
        p=executor.submit(obj,[arx_added[0,indx],arx_added[1,indx],arx_added[2,indx],arx_added[3,indx]],wind_index[indx])
        #print(p.result())
        processes.append(p)
    for f in concurrent.futures.as_completed(processes):
        print(f.result())

print('-------------------------------')
print(processes[0].result())
print(processes[-1].result())
#processes=[]
"""
for indx in range(60):
    p=multiprocessing.Process(target=obj,args=[arx_added[0,indx],arx_added[1,indx],arx_added[2,indx],arx_added[3,indx],wind_index[indx]])
    p.start()
    processes.append(p)
    
for process in processes:
    process.join()    """