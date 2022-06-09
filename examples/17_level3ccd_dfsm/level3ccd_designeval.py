'''
Usage:
    python level3ccd_designeval.py DB_PATH ID_NUM [EVAL=True] [PRINT=False] [SAVE=False]
Example:
    python level3ccd_designeval.py output/linear_data.db 1
    python level3ccd_designeval.py output/linear_data.db 1 True
    python level3ccd_designeval.py output/linear_data.db 1 True False
    python level3ccd_designeval.py output/linear_data.db 1 True False False
'''

import os
import sys
import yaml
from level3ccd_class import turbine_design
from level3ccd_class import sql_design

EVAL = True
PRINT = False
SAVE = False

if (len(sys.argv) < 3) or (len(sys.argv) > 6):
    raise ValueError('Wrong number of arguments. Usage: python level3ccd_designeval.py DB_PATH ID_NUM [EVAL=True] [PRINT=False]')
else:
    dbpath = sys.argv[1]
    idnum = int(sys.argv[2])
    if len(sys.argv) > 3:
        if sys.argv[3].lower() == 'false':
            EVAL = False
    if len(sys.argv) > 4:
        if sys.argv[4].lower() == 'true':
            PRINT = True
    if len(sys.argv) > 5:
        if sys.argv[5].lower() == 'true':
            SAVE = True

db = sql_design(dbpath = dbpath)
db.create_connection()
des, par = db.get_design_dict(idnum)
db.close_connection()

wt = turbine_design()
wt.design_SN = idnum
wt.design = des
wt.param = par

if PRINT:
    print('fixed parameter values = {:}'.format(par))
    print('design parameter values = {:}'.format(des))

if not EVAL:
    sys.exit(os.EX_OK)

wt.create_turbine()

if SAVE:
    savedir = os.path.dirname(dbpath)
    with open(os.path.join(savedir, 'modeling_options_{:08d}.yaml'.format(idnum)), 'wt') as yml:
        yaml.dump(wt.modeling_options, yml)
    with open(os.path.join(savedir, 'analysis_options_{:08d}.yaml'.format(idnum)), 'wt') as yml:
        yaml.dump(wt.analysis_options, yml)
    with open(os.path.join(savedir, 'turbine_model_{:08d}.yaml'.format(idnum)), 'wt') as yml:
        yaml.dump(wt.turbine_model, yml)

# Visualize or Compute cost only for debug purpose
# wt.visualize_turbine()
# wt.compute_cost_only()

wt.compute_full_model(OF_run_dir=os.path.join(os.path.dirname(dbpath), 'OF_lin'))

wt_linear_result = wt.linear
wt_linear_result['cost_per_year'] = float(wt.cost_per_year)
wt_linear_result['design_life_year'] = float(wt.design_life_year)

savedir = os.path.dirname(dbpath)
savefilepath = os.path.join(savedir, 'result{:08d}.yaml'.format(idnum))
with open(savefilepath, 'wt') as yml:
    yaml.dump(wt_linear_result, yml)

sys.exit(os.EX_OK)

