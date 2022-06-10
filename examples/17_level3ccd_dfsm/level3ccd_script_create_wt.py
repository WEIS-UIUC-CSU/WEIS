import os, time, yaml
from wisdem.commonse.mpi_tools  import MPI
from level3ccd_class import turbine_design

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0

if rank == 0:
    run_dir = os.path.dirname(os.path.realpath(__file__))
    fname_wt_input = os.path.abspath(os.path.join(os.path.dirname(run_dir), '06_IEA_15-240-RWT', 'IEA-15-240-RWT.yaml'))
    fname_modeling_options = os.path.abspath(os.path.join(run_dir, 'modeling_options.yaml'))
    fname_analysis_options = os.path.abspath(os.path.join(run_dir, 'analysis_options.yaml'))
    # Start
    tt = time.time()
    # Design variable values and other parameters
    des = {
        'hub_height': 150.0,
        'tower_top_diameter': 6.5,
        'tower_bottom_thickness': 0.041058,
        'tower_top_thickness': 0.020826,
        'rotor_diameter': 240.0}
    par = {
        'tower_div': 11,
        'tower_bottom_height': 15.0,
        'tower_bottom_diameter': 10.0,
        'tower_yaw_thickness': 0.023998,
        'monopile_bottom_height': -75.0,
        'water_depth': 30.0}
    # Create turbine
    wt = turbine_design()
    wt.design = des
    wt.param = par
    wt.create_turbine()
    # Save
    with open(os.path.join(run_dir, 'modeling_options_created.yaml'), 'wt') as yml:
        yaml.dump(wt.modeling_options, yml)
    with open(os.path.join(run_dir, 'analysis_options_created.yaml'), 'wt') as yml:
        yaml.dump(wt.analysis_options, yml)
    with open(os.path.join(run_dir, 'turbine_model_created.yaml'), 'wt') as yml:
        yaml.dump(wt.turbine_model, yml)
    # Calculate cost
    wt.compute_cost_only()
    # Visualize
    wt.visualize_turbine()
    # Elapsed time
    print('Run time: %f'%(time.time()-tt))





