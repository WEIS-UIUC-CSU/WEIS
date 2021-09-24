import os
import sys
import time
start_time = time.time()
import numpy as np
import subprocess
from level3ccd_class import (sql_design, surrogate_model)

dbpath = 'output/linear_data.db'
max_cores = 1

if len(sys.argv) > 1:
    if sys.argv[1].lower() == '-np':
        max_cores = int(sys.argv[2])

if len(sys.argv) == 5:
    i_start = int(sys.argv[3])
    i_end = int(sys.argv[4])
else:
    i_start = 1
    i_end = 9999999999

d = sql_design(dbpath=dbpath)
d.create_connection()
total_design_id_list = d.get_design_id()
d.cursor.close()
d.conn.close()

processes = set()

for id_val in total_design_id_list:
    if (id_val < i_start) or (id_val > i_end):
        continue
    processes.add(subprocess.Popen(['python', 'level3ccd_designeval.py', dbpath, str(id_val)]))
    if len(processes) >= max_cores:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])

end_time = time.time()
print('Time elapsed = {:} seconds'.format(end_time - start_time))
