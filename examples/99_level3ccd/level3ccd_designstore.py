'''
Usage:
    python level3ccd_designstore.py DB_PATH ID_NUM
Example:
    python level3ccd_designstore.py output/linear_data.db 1
'''

import os
import sys
import yaml
from level3ccd_class import turbine_design
from level3ccd_class import sql_design

if len(sys.argv) != 3:
    raise ValueError('Wrong number of arguments. Usage: python level3ccd_designeval.py DB_PATH ID_NUM [EVAL=True] [PRINT=False]')
else:
    dbpath = sys.argv[1]
    id_val = int(sys.argv[2])

db = sql_design(dbpath = dbpath)
db.create_connection()
total_design_id_list = db.get_design_id()
if id_val not in total_design_id_list:
    raise ValueError('ID_NUM = {:} not defined in the design DB file.'.format(id_val))
des, par = db.get_design_dict(id_val)
db.close_connection()

ympath = os.path.dirname(dbpath) # ./output/
ofpath = os.path.join(ympath, 'OF_lin') # ./output/OF_lin/
linpath = os.path.join(ofpath, '{:08d}'.format(id_val)) # ./output/OF_lin/00000001/

if os.path.isdir(linpath):
    # Get filenames for linearization results
    linflist = []
    linflistall = []
    for fname in os.listdir(linpath):
        if fname.lower().endswith('.lin'):
            linflistall.append(fname)
            fname_base = os.path.splitext(os.path.splitext(fname)[0])[0]
            if fname_base not in linflist:
                linflist.append(fname_base)
    linflist.sort()
    linflistall.sort()
    
    if len(linflist) > 0:
        wt = turbine_design()
        wt.design_SN = id_val
        wt.design = des
        wt.param = par
        wt.create_turbine()
        wind_speeds = wt.modeling_options['Level2']['linearization']['wind_speeds']
        NLinTimes = wt.modeling_options['Level2']['linearization']['NLinTimes']
        if len(wind_speeds) == len(linflist):
            if len(wind_speeds)*NLinTimes == len(linflistall):
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
                print('WARNING, path {:} has missing .lin files.'.format(os.path.join(ofpath, '{:08d}'.format(id_val))))
        else:
            print('WARNING, path {:} has missing wind speeds.'.format(os.path.join(ofpath, '{:08d}'.format(id_val))))
    else:
        print('WARNING, unable to find .lin files from path {:}.'.format(os.path.join(ofpath, '{:08d}'.format(id_val))))
else:
    print('WARNING, path {:} does not exist.'.format(os.path.join(ofpath, '{:08d}'.format(id_val))))
