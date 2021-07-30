from wisdem.commonse.mpi_tools  import MPI
import os, time
import weis.inputs as sch
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

def visualize_tower(wt):
    rotor_diameter = wt['assembly']['rotor_diameter']
    hub_height = wt['assembly']['hub_height']
    tower_z_grid = wt["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["grid"]
    tower_z_values = wt["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["values"]
    tower_outer_diameter_grid = wt['components']['tower']['outer_shape_bem']['outer_diameter']['grid']
    tower_outer_diameter_values = wt['components']['tower']['outer_shape_bem']['outer_diameter']['values']
    print('rotor diameter = {:}'.format(rotor_diameter))
    print('hub height = {:}'.format(hub_height))
    print('tower height = {:}'.format(tower_z_values[-1]))
    print('tower outer diameter at bottom = {:}'.format(tower_outer_diameter_values[0]))
    print('tower outer diameter at top = {:}'.format(tower_outer_diameter_values[-1]))

run_dir = os.path.dirname( os.path.realpath(__file__) ) + os.sep
wisdem_examples = os.path.join(os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ), "WISDEM", "examples")
fname_wt_input = os.path.join(wisdem_examples, "02_reference_turbines", "IEA-15-240-RWT.yaml")
fname_modeling_options = run_dir + "modeling_options.yaml"
fname_analysis_options = run_dir + "analysis_options.yaml"

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0

tt = time.time()

p_tower_div = 12
p_tower_bottom_height = 15.0
p_tower_bottom_diameter = 10.0
d_hub_height = 150.0
d_tower_top_diameter = 6.5
d_rotor_diameter = 242.25

wt = sch.load_geometry_yaml(fname_wt_input)
tower_grd = wt["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["grid"]
tower_val = wt["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["values"]
tower_dia = wt['components']['tower']['outer_shape_bem']['outer_diameter']['values']
hub_val = wt['assembly']['hub_height']
tower_top_to_hub_height = hub_val - tower_val[-1]
tower_top_height = d_hub_height - tower_top_to_hub_height

tower_zcoord_new = np.linspace(p_tower_bottom_height, tower_top_height, p_tower_div)
for tidx in range(0, tower_zcoord_new.shape[0]-1):
    tower_zcoord_new = np.insert(
        tower_zcoord_new,
        2*tidx + 1,
        tower_zcoord_new[2*tidx] + 0.01
    )
tower_grd_new = np.interp(
    tower_zcoord_new,
    [tower_zcoord_new[0], tower_zcoord_new[-1]],
    [0.0, 1.0]
)
tower_diameter_new = p_tower_bottom_diameter*np.ones(tower_zcoord_new.shape, dtype=float)
tower_diameter_interp1d = interp1d(tower_val, tower_dia, kind='cubic')
tower_diameter_fill = tower_diameter_interp1d(np.linspace(tower_zcoord_new[3], tower_top_height, p_tower_div - 1))
for tidx in range(4, tower_diameter_new.shape[0]):
    if np.remainder(tidx, 2) == 0:
        fidx = int(tidx/2) - 1
        tower_diameter_new[tidx] = tower_diameter_fill[fidx]
    elif np.remainder(tidx, 2) == 1:
        fidx = int((tidx-1)/2) - 1
        tower_diameter_new[tidx] = tower_diameter_fill[fidx]

wt['assembly']['rotor_diameter'] = d_rotor_diameter
wt['assembly']['hub_height'] = d_hub_height
wt["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["grid"] = tower_grd_new.tolist()
wt["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["values"] = tower_zcoord_new.tolist()
wt['components']['tower']['outer_shape_bem']['outer_diameter']['grid'] = tower_grd_new.tolist()
wt['components']['tower']['outer_shape_bem']['outer_diameter']['values'] = tower_diameter_new.tolist()

sch.write_geometry_yaml(wt, 'test_turbine.yaml')

if rank == 0:
    visualize_tower(wt)
    print('Run time: %f'%(time.time()-tt))