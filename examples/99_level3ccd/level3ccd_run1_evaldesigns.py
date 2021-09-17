import os
import sys
import time
start_time = time.time()
import numpy as np
import subprocess
from level3ccd_class import (sql_design, surrogate_model)

# Define design problem

N = 100                 # number of sample points
max_cores = 1
random_state = 0
xlimits = np.array([
    [145.0, 155.0],     # hub_height
    [5.9, 7.1],         # tower_top_diameter
    [0.035, 0.45],       # tower_bottom_thickness
    [0.015, 0.025],     # tower_top_thickness
    [230.0, 250.0]      # rotor_diameter
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

# Begin DFSM script

if len(sys.argv) > 1:
    if sys.argv[1].lower() == '-np':
        max_cores = int(sys.argv[2])

d = sql_design(dbpath=dbpath)
d.create_connection()
total_design_id_list = d.get_design_id()

d.cursor.close()
d.conn.close()

processes = set()

for id_val in total_design_id_list:
    processes.add(subprocess.Popen(['python', 'level3ccd_designeval.py', dbpath, str(id_val)]))
    if len(processes) >= max_cores:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])

end_time = time.time()
print('Time elapsed = {:} seconds'.format(end_time - start_time))
