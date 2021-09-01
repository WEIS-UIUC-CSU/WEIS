from wisdem.commonse.mpi_tools  import MPI
from warnings import warn
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
    tower_thickness_grid = wt["components"]["tower"]['internal_structure_2d_fem']['layers'][0]['thickness']['grid']
    tower_thickness_values = wt["components"]["tower"]['internal_structure_2d_fem']['layers'][0]['thickness']['values']
    if (
            np.allclose(tower_z_grid, tower_outer_diameter_grid) and
            np.allclose(tower_z_grid, tower_thickness_grid)
    ):
        print('tower grid consistency test passed')
    else:
        warn('tower grid consistency test failed')
    print('rotor diameter = {:}'.format(rotor_diameter))
    print('hub height = {:}'.format(hub_height))
    print('tower height = {:}'.format(tower_z_values[-1]))
    print('tower outer diameter at bottom = {:}'.format(tower_outer_diameter_values[0]))
    print('tower outer diameter at top = {:}'.format(tower_outer_diameter_values[-1]))
    print('tower thickness at bottom = {:}'.format(tower_thickness_values[0]))
    print('tower thickness at top = {:}'.format(tower_thickness_values[-3]))
    print('tower thickness at yaw control device = {:}'.format(tower_thickness_values[-1]))

    plt.plot(tower_z_values, tower_thickness_values)
    plt.show()

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

# Parameters
p_tower_div = 11
p_tower_bottom_height = 15.0
p_tower_bottom_diameter = 10.0
p_tower_top_thickness = 0.023998

# Design variables
d_hub_height = 150.0
d_tower_top_diameter = 6.5
d_tower_bottom_thickness = 0.041058
d_tower_top_thickness = 0.020826
d_rotor_diameter = 240

wt = sch.load_geometry_yaml(fname_wt_input)
tower_grd = wt["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["grid"]
tower_val = wt["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["values"]
tower_dia = wt['components']['tower']['outer_shape_bem']['outer_diameter']['values']
tower_thk = wt["components"]["tower"]['internal_structure_2d_fem']['layers'][0]['thickness']['values']
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
tower_diameter_interp1d = interp1d(
    p_tower_bottom_height + (np.array(tower_val) - tower_val[0])*(tower_zcoord_new[-1] - tower_zcoord_new[0])/(tower_val[-1] - tower_val[0]),
    tower_dia,
    kind='cubic'
)
tower_diameter_fill = tower_diameter_interp1d(np.linspace(tower_zcoord_new[3], tower_top_height, p_tower_div - 1))
for tidx in range(4, tower_diameter_new.shape[0]):
    if np.remainder(tidx, 2) == 0:
        fidx = int(tidx/2) - 1
        tower_diameter_new[tidx] = tower_diameter_fill[fidx]
    elif np.remainder(tidx, 2) == 1:
        fidx = int((tidx-1)/2) - 1
        tower_diameter_new[tidx] = tower_diameter_fill[fidx]
tower_thk = np.array(tower_thk)
tower_thk = (tower_thk - tower_thk[0])/(tower_thk[-3] - tower_thk[0])
tower_thk = d_tower_bottom_thickness + (d_tower_top_thickness - d_tower_bottom_thickness)*tower_thk
tower_thickness_interp1d = interp1d(
    np.array([tower_grd[i] for i in range(0,len(tower_grd)-1,2)])/tower_grd[len(tower_grd)-3],
    [tower_thk[i] for i in range(0,len(tower_grd)-1,2)]
)
tower_thickness_new = d_tower_bottom_thickness*np.ones(tower_zcoord_new.shape, dtype=float)
for tidx in range(1, tower_thickness_new.shape[0]-2, 2):
    tower_thickness_new[tidx + 1] = tower_thickness_interp1d(tower_grd_new[tidx + 1]/tower_grd_new[len(tower_grd_new)-3])
    tower_thickness_new[tidx] = tower_thickness_new[tidx + 1]
tower_thickness_new[tower_thickness_new.shape[0]-2] = p_tower_top_thickness
tower_thickness_new[tower_thickness_new.shape[0]-1] = p_tower_top_thickness

wt['assembly']['rotor_diameter'] = d_rotor_diameter
wt['assembly']['hub_height'] = d_hub_height
wt["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["grid"] = tower_grd_new.tolist()
wt["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["values"] = tower_zcoord_new.tolist()
wt['components']['tower']['outer_shape_bem']['outer_diameter']['grid'] = tower_grd_new.tolist()
wt['components']['tower']['outer_shape_bem']['outer_diameter']['values'] = tower_diameter_new.tolist()
wt["components"]["tower"]['internal_structure_2d_fem']['reference_axis']['z']['grid'] = tower_grd_new.tolist()
wt["components"]["tower"]['internal_structure_2d_fem']['reference_axis']['z']['values'] = tower_zcoord_new.tolist()
wt["components"]["tower"]['internal_structure_2d_fem']['layers'][0]['thickness']['grid'] = tower_grd_new.tolist()
wt["components"]["tower"]['internal_structure_2d_fem']['layers'][0]['thickness']['values'] = tower_thickness_new.tolist()

sch.write_geometry_yaml(wt, 'test_turbine.yaml')

if rank == 0:
    visualize_tower(wt)
    print('Run time: %f'%(time.time()-tt))
