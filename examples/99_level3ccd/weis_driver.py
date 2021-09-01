#import os
#from weis.glue_code.runWEIS     import run_weis

#import numpy as np
import os, sys #, time
import openmdao.api as om

from weis.glue_code.gc_LoadInputs     import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_WT_InitModel import yaml2openmdao
from weis.glue_code.gc_PoseOptimization  import PoseOptimizationWEIS
from weis.glue_code.glue_code         import WindPark
from wisdem.commonse.mpi_tools        import MPI
from wisdem.commonse                  import fileIO
from weis.glue_code.gc_ROSCOInputs    import assign_ROSCO_values

if MPI:
    from wisdem.commonse.mpi_tools import map_comm_heirarchical, subprocessor_loop, subprocessor_stop
    max_cores = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    color_i = 0
else:
    max_cores = 1
    rank = 0
    color_i = 0

#mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
mydir = '/Users/yonghoonlee/Dropbox/ATLANTIS_WEIS/WEIS/examples/99_level3ccd'
fname_wt_input         = mydir + os.sep + "test_turbine.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

#wt_opt, modeling_options, analysis_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)

wt_initial = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_analysis_options)
wt_init, modeling_options, analysis_options = wt_initial.get_input_data()
myopt = PoseOptimizationWEIS(wt_init, modeling_options, analysis_options)

# Solve OpenFAST in serial, populations in parallel
n_OF_runs_parallel = 1
max_parallel_OF_runs = 1
n_POP_parallel = max_cores
comm_map_up = comm_map_down = {}
for r in range(max_cores):
    comm_map_up[r] = [r]

folder_output = analysis_options['general']['folder_output']
if rank == 0:
    if not os.path.isdir(folder_output):
        os.makedirs(folder_output)

modeling_options['General']['openfast_configuration']['mpi_run'] = False
modeling_options['General']['openfast_configuration']['cores']   = 1
modeling_options['General']['openfast_configuration']['OF_run_dir'] += (os.sep + 'rank_' + str(rank))
wt_opt = om.Problem(model=WindPark(modeling_options = modeling_options, opt_options = analysis_options))
wt_opt = myopt.set_recorders(wt_opt)
wt_opt.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs','totals']

wt_opt.setup(derivatives=False)

# Load initial wind turbine data from wt_initial to the openmdao problem
wt_opt = yaml2openmdao(wt_opt, modeling_options, wt_init, analysis_options)
wt_opt = assign_ROSCO_values(wt_opt, modeling_options, wt_init['control'])
wt_opt = myopt.set_initial(wt_opt, wt_init)
wt_opt = myopt.set_restart(wt_opt)
sys.stdout.flush()

# Run OpenMDAO problem
wt_opt.run_model()

# Save data
froot_out = os.path.join(folder_output, analysis_options['general']['fname_output']) + (os.sep + 'rank_' + str(rank))
wt_initial.update_ontology_control(wt_opt)
modeling_options['General']['openfast_configuration']['fst_vt'] = {}
wt_initial.write_ontology(wt_opt, froot_out)
wt_initial.write_options(froot_out)

# Save data to numpy and matlab arrays
fileIO.save_data(froot_out, wt_opt)
