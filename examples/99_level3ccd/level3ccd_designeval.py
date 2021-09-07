'''
Usage:
    python level3ccd_designeval.py DB_PATH ID_NUM
Example:
    python level3ccd_designeval.py temp/linear_data.db 1
'''

import os
import sys
import yaml
import numpy as np
from level3ccd_class import turbine_design
from level3ccd_class import sql_design

if len(sys.argv) != 3:
    dbpath = '/Users/yonghoonlee/Dropbox/ATLANTIS_WEIS/WEIS/examples/99_level3ccd/temp/linear_data.db'
    idnum = 1
    #raise ValueError('Wrong number of arguments. Usage: python level3ccd_designeval.py DB_PATH ID_NUM')
else:
    dbpath = sys.argv[1]
    idnum = int(sys.argv[2])

db = sql_design(dbpath = dbpath)
db.create_connection()
des, par = db.get_design_dict(idnum)
db.close_connection()

wt = turbine_design()
wt.design_SN = idnum
wt.design = des
wt.param = par
wt.create_turbine()
#wt.visualize_turbine()
#wt.compute_cost_only()
wt.compute_full_model()

wt_linear_result = wt.linear
wt_linear_result['cost_per_year'] = float(wt.cost_per_year)
wt_linear_result['design_life_year'] = float(wt.design_life_year)

savedir = os.path.dirname(dbpath)
savefilepath = os.path.join(savedir, 'result{:08d}.yaml'.format(idnum))
with open(savefilepath, 'wt') as yml:
    yaml.dump(wt_linear_result, yml)

sys.exit(os.EX_OK)