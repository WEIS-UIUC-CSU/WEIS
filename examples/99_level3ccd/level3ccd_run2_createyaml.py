import os
import sys
import time
start_time = time.time()
import numpy as np
import yaml

from level3ccd_class import (turbine_design, sql_design, surrogate_model)

dbpath = 'output/linear_data.db'
ympath = os.path.dirname(dbpath)
ofpath = os.path.join(ympath, 'OF_lin')

d = sql_design(dbpath=dbpath)
d.create_connection()
total_design_id_list = d.get_design_id()

for id_val in total_design_id_list:
    linpath = os.path.join(ofpath, '{:08d}'.format(id_val))
    if os.path.isdir(linpath):
        # Get filenames for linearization results
        linflist = []
        for fname in os.listdir(linpath):
            if fname.lower().endswith('.lin'):
                fname_base = os.path.splitext(os.path.splitext(fname)[0])[0]
                if fname_base not in linflist:
                    linflist.append(fname_base)
        linflist.sort()
        
        if len(linflist) > 0:
            des, par = d.get_design_dict(id_val)
            wt = turbine_design()
            wt.design_SN = id_val
            wt.design = des
            wt.param = par
            wt.create_turbine()
            wt.compute_cost_only()
            wt.save_linear_model(FAST_runDirectory=linpath, lin_case_name=linflist)

            wt_linear_result = None
            wt_linear_result = wt.linear
            wt_linear_result['cost_per_year'] = float(wt.cost_per_year)
            wt_linear_result['design_life_year'] = float(wt.design_life_year)
            wt_linear_result['design'] = des
            wt_linear_result['parameter'] = par

            with open(os.path.join(ympath, '{:08d}.yaml'.format(id_val)), 'wt') as yml:
                yaml.safe_dump(wt_linear_result, yml)

    else:
        print('WARNING, path {:} does not exist.'.format(os.path.join(ofpath, '{:08d}'.format(id_val))))

d.cursor.close()
d.conn.close()

