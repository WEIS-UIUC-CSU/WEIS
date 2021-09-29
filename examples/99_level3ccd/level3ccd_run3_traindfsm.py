import numpy as np
import os
import pickle
import time
start_time = time.time()
from level3ccd_class import dfsm_class

dbpath = 'output/linear_data.db'

dfsm = dfsm_class()
dfsm.load_linear_models(dbpath=dbpath)
dfsm.surrogate_model = 'KRG'
dfsm.train_sm()
with open(os.path.join(os.path.dirname(dbpath), 'dfsm.pkl'), 'wb') as pkl:
    pickle.dump(dfsm, pkl)

end_time = time.time()
print('Time elapsed for training surrogate models: {:} seconds'.format(end_time - start_time))
