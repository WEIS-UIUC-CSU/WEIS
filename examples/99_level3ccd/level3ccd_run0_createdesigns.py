import os
import sys
import time
start_time = time.time()
import numpy as np
import subprocess
from level3ccd_class import (sql_design, surrogate_model)

# Define design problem

N = 400                 # number of sample points
max_cores = 1
random_state = 0
xlimits = np.array([
    [140.0, 160.0],     # hub_height
    [5.5, 7.5],         # tower_top_diameter
    [0.030, 0.050],       # tower_bottom_thickness
    [0.010, 0.030],     # tower_top_thickness
    [220.0, 260.0]      # rotor_diameter
])
param = {
    'tower_div': 11,
    'tower_bottom_height': 15.0,
    'tower_bottom_diameter': 10.0,
    'tower_yaw_thickness': 0.023998,
    'monopile_bottom_height': -75.0,
    'water_depth': 30.0
}
dbpath = 'output/linear_data.db'

sm = surrogate_model()
x_smp = sm.sampling(N, xlimits, criterion='ese', random_state=random_state, extreme=True)
n_smp, n_dim = x_smp.shape

d = sql_design(dbpath=dbpath)
d.remove_db()
for idx in range(n_smp):
    design = {
        'hub_height': x_smp[idx, 0],
        'tower_top_diameter': x_smp[idx, 1],
        'tower_bottom_thickness': x_smp[idx, 2],
        'tower_top_thickness': x_smp[idx, 3],
        'rotor_diameter': x_smp[idx, 4]
    }
    d.add_data(design, param)
d.conn.commit()

total_design_id_list = d.get_design_id()

d.cursor.close()
d.conn.close()

end_time = time.time()
print('Time elapsed = {:} seconds'.format(end_time - start_time))
