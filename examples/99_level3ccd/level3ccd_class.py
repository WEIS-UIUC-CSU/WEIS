import os
import sys
import pickle
import numpy as np
import openmdao.api as om
import weis

from weis.glue_code.gc_LoadInputs           import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_WT_InitModel       import yaml2openmdao
from weis.glue_code.gc_PoseOptimization     import PoseOptimizationWEIS
from weis.glue_code.glue_code               import WindPark
from weis.glue_code.gc_ROSCOInputs          import assign_ROSCO_values
#from weis.aeroelasticse.FAST_reader         import InputReader_OpenFAST
#from wisdem.commonse                        import fileIO
from wisdem.glue_code.gc_PoseOptimization   import PoseOptimization as PoseOptimizationWISDEM

from scipy.interpolate                      import interp1d
from matplotlib                             import pyplot as plt
from matplotlib.patches                     import (Rectangle, Circle)
from copy                                   import deepcopy

class turbine_design:

    def __init__(self, **kwargs):

        self.weis_root = os.path.dirname(os.path.dirname(os.path.abspath(weis.__file__)))
        self.run_path = os.path.dirname(os.path.abspath(__file__))

        if 'wt_name' in kwargs.keys():
            wt_name = kwargs['wt_name']
        else:
            wt_name = 'IEA-15-240-RWT.yaml'

        fname_wt_input = os.path.join(self.weis_root, 'WISDEM', 'examples', '02_reference_turbines', wt_name)
        fname_modeling = os.path.join(self.run_path, 'modeling_options.yaml')
        fname_analysis = os.path.join(self.run_path, 'analysis_options.yaml')
        self.wt_ontology = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling, fname_analysis)
        
        turbine_model, modeling_options, analysis_options = self.wt_ontology.get_input_data()
        self.turbine_model = turbine_model
        self.modeling_options = modeling_options
        self.analysis_options = analysis_options

        self.design = dict()
        self.param = dict()
        self.design_SN = 0
        self.result = None

        # Store reference turbine values
        self._ref_tower_grd = self.turbine_model['components']['tower']['outer_shape_bem']['reference_axis']['z']['grid']
        self._ref_tower_val = self.turbine_model['components']['tower']['outer_shape_bem']['reference_axis']['z']['values']
        self._ref_tower_dia = self.turbine_model['components']['tower']['outer_shape_bem']['outer_diameter']['values']
        self._ref_tower_thk = self.turbine_model['components']['tower']['internal_structure_2d_fem']['layers'][0]['thickness']['values']
        self._ref_hub_val = self.turbine_model['assembly']['hub_height']
        self._ref_tower_top_to_hub_height = self._ref_hub_val - self._ref_tower_val[-1]
        self._ref_monopile_grd = self.turbine_model['components']['monopile']['outer_shape_bem']['reference_axis']['z']['grid']
        self._ref_monopile_val = self.turbine_model['components']['monopile']['outer_shape_bem']['reference_axis']['z']['values']
        self._ref_monopile_dia = self.turbine_model['components']['monopile']['outer_shape_bem']['outer_diameter']['values']
        self._ref_water_depth = self.turbine_model['environment']['water_depth']

        # Default values

        self._p_tower_div_default = 11
        self._p_tower_bottom_height_default = 15.0
        self._p_tower_bottom_diameter_default = 10.0
        self._p_tower_top_thickness_default = 0.023998
        self._p_monopile_bottom_height_default = -75.0
        self._p_water_depth_default = 30.0

        self._d_hub_height_default = 150.0
        self._d_tower_top_diameter_default = 6.5
        self._d_tower_bottom_thickness_default = 0.041058
        self._d_tower_top_thickness_default = 0.020826
        self._d_rotor_diameter_default = 240

    def create_turbine(self):

        # Check if design dictionary is empty
        if len(self.design.keys()) == 0:
            raise ValueError('Design is not specified.')

        # Obtaining required parameters
        # Tower division
        if 'tower_div' in self.param.keys():
            p_tower_div = self.param['tower_div']
        else:
            p_tower_div = self._p_tower_div_default
        # Tower bottom height
        if 'tower_bottom_height' in self.param.keys():
            p_tower_bottom_height = self.param['tower_bottom_height']
        else:
            p_tower_bottom_height = self._p_tower_bottom_height_default
        # Tower bottom diameter
        if 'tower_bottom_diameter' in self.param.keys():
            p_tower_bottom_diameter = self.param['tower_bottom_diameter']
        else:
            p_tower_bottom_diameter = self._p_tower_bottom_diameter_default
        # Tower top thickness
        if 'tower_top_thickness' in self.param.keys():
            p_tower_top_thickness = self.param['tower_top_thickness']
        else:
            p_tower_top_thickness = self._p_tower_top_thickness_default
        # Monopile bottom height
        if 'monopile_bottom_height' in self.param.keys():
            p_monopile_bottom_height = self.param['monopile_bottom_height']
        else:
            p_monopile_bottom_height = self._p_monopile_bottom_height_default
        # Water depth
        if 'water_depth' in self.param.keys():
            p_water_depth = self.param['water_depth']
        else:
            p_water_depth = self._p_water_depth_default

        # Updating required parameter dictionary
        self.param = {
            'tower_div': p_tower_div,
            'tower_bottom_height': p_tower_bottom_height,
            'tower_bottom_diameter': p_tower_bottom_diameter,
            'tower_top_thickness': p_tower_top_thickness,
            'monopile_bottom_height': p_monopile_bottom_height,
            'water_depth': p_water_depth
        }

        # Obtaining design parameter values
        # Hub height
        if 'hub_height' in self.design.keys():
            d_hub_height = self.design['hub_height']
        else:
            d_hub_height = self._d_hub_height_default
        # Tower top diameter
        if 'tower_top_diameter' in self.design.keys():
            d_tower_top_diameter = self.design['tower_top_diameter']
        else:
            d_tower_top_diameter = self._d_tower_top_diameter_default
        # Tower bottom thickness
        if 'tower_bottom_thickness' in self.design.keys():
            d_tower_bottom_thickness = self.design['tower_bottom_thickness']
        else:
            d_tower_bottom_thickness = self._d_tower_bottom_thickness_default
        # Tower top thickness
        if 'tower_top_thickness' in self.design.keys():
            d_tower_top_thickness = self.design['tower_top_thickness']
        else:
            d_tower_top_thickness = self._d_tower_top_thickness_default
        # Rotor diameter
        if 'rotor_diameter' in self.design.keys():
            d_rotor_diameter = self.design['rotor_diameter']
        else:
            d_rotor_diameter = self._d_rotor_diameter_default

        # Updating design parameter values dictionary
        self.design = {
            'hub_height': d_hub_height,
            'tower_top_diameter': d_tower_top_diameter,
            'tower_bottom_thickness': d_tower_bottom_thickness,
            'tower_top_thickness': d_tower_top_thickness,
            'rotor_diameter': d_rotor_diameter,
        }

        # Derived quantity
        tower_top_height = d_hub_height - self._ref_tower_top_to_hub_height
        tower_zcoord_new = np.linspace(p_tower_bottom_height, tower_top_height, p_tower_div)
        for tidx in range(0, tower_zcoord_new.shape[0] - 1):
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
            p_tower_bottom_height \
                + (np.array(self._ref_tower_val) - self._ref_tower_val[0]) \
                    *(tower_zcoord_new[-1] - tower_zcoord_new[0]) \
                    /(self._ref_tower_val[-1] - self._ref_tower_val[0]),
            p_tower_bottom_diameter \
                + (np.array(self._ref_tower_dia) - self._ref_tower_dia[0]) \
                    *(d_tower_top_diameter - p_tower_bottom_diameter) \
                    /(self._ref_tower_dia[-1] - self._ref_tower_dia[0]),
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
        tower_thk = np.array(self._ref_tower_thk)
        tower_thk = (tower_thk - tower_thk[0])/(tower_thk[-3] - tower_thk[0])
        tower_thk = d_tower_bottom_thickness + (d_tower_top_thickness - d_tower_bottom_thickness)*tower_thk
        tower_thickness_interp1d = interp1d(
            np.array(
                [self._ref_tower_grd[i] for i in range(0,len(self._ref_tower_grd)-1,2)]
            )/self._ref_tower_grd[len(self._ref_tower_grd)-3],
            [tower_thk[i] for i in range(0,len(self._ref_tower_grd)-1,2)]
        )
        tower_thickness_new = d_tower_bottom_thickness*np.ones(tower_zcoord_new.shape, dtype=float)
        for tidx in range(1, tower_thickness_new.shape[0]-2, 2):
            tower_thickness_new[tidx + 1] = tower_thickness_interp1d(tower_grd_new[tidx + 1]/tower_grd_new[len(tower_grd_new)-3])
            tower_thickness_new[tidx] = tower_thickness_new[tidx + 1]
        tower_thickness_new[tower_thickness_new.shape[0]-2] = p_tower_top_thickness
        tower_thickness_new[tower_thickness_new.shape[0]-1] = p_tower_top_thickness
        monopile_z_coord_new = np.interp(
            self._ref_monopile_grd,
            [0.0, 1.0],
            [p_monopile_bottom_height, p_tower_bottom_height]
        )

        self.turbine_model['assembly']['rotor_diameter'] = d_rotor_diameter
        self.turbine_model['assembly']['hub_height'] = d_hub_height
        self.turbine_model['components']['tower']['outer_shape_bem']['reference_axis']['z']['grid'] = tower_grd_new.tolist()
        self.turbine_model['components']['tower']['outer_shape_bem']['reference_axis']['z']['values'] = tower_zcoord_new.tolist()
        self.turbine_model['components']['tower']['outer_shape_bem']['outer_diameter']['grid'] = tower_grd_new.tolist()
        self.turbine_model['components']['tower']['outer_shape_bem']['outer_diameter']['values'] = tower_diameter_new.tolist()
        self.turbine_model['components']['tower']['internal_structure_2d_fem']['reference_axis']['z']['grid'] = tower_grd_new.tolist()
        self.turbine_model['components']['tower']['internal_structure_2d_fem']['reference_axis']['z']['values'] = tower_zcoord_new.tolist()
        self.turbine_model['components']['tower']['internal_structure_2d_fem']['layers'][0]['thickness']['grid'] = tower_grd_new.tolist()
        self.turbine_model['components']['tower']['internal_structure_2d_fem']['layers'][0]['thickness']['values'] = tower_thickness_new.tolist()
        self.turbine_model['components']['monopile']['outer_shape_bem']['reference_axis']['z']['values'] = monopile_z_coord_new.tolist()
        self.turbine_model['components']['monopile']['outer_shape_bem']['outer_diameter']['values'] = (np.ones(shape=(len(self._ref_monopile_dia)), dtype=float)*p_tower_bottom_diameter).tolist()
        self.turbine_model['components']['monopile']['internal_structure_2d_fem']['reference_axis']['z']['values'] = monopile_z_coord_new.tolist()
        self.turbine_model['environment']['water_depth'] = p_water_depth
        
        self.design_SN += 1
        
    def compute_cost_only(self):
        
        turbine_model = deepcopy(self.turbine_model)
        modeling_options = deepcopy(self.modeling_options)
        analysis_options = deepcopy(self.analysis_options)
        
        modeling_options['Level1']['flag'] = False
        modeling_options['Level2']['flag'] = False
        modeling_options['Level3']['flag'] = False
        modeling_options['ROSCO']['flag'] = False
        analysis_options['recorder']['flag'] = False
        
        myopt = PoseOptimizationWISDEM(turbine_model, modeling_options, analysis_options)
        wt_opt = om.Problem(
            model=WindPark(modeling_options=modeling_options, opt_options=analysis_options)
        )
        wt_opt.setup()
        wt_opt = yaml2openmdao(wt_opt, modeling_options, turbine_model, analysis_options)
        wt_opt = myopt.set_initial(wt_opt, turbine_model)
        wt_opt.run_model()
        
        self.result = wt_opt
    
    def compute_full_model(self):
        
        turbine_model = deepcopy(self.turbine_model)
        modeling_options = deepcopy(self.modeling_options)
        analysis_options = deepcopy(self.analysis_options)
        
        modeling_options['Level1']['flag'] = False
        modeling_options['Level2']['flag'] = True
        modeling_options['Level3']['flag'] = False
        modeling_options['ROSCO']['flag'] = True
        analysis_options['recorder']['flag'] = False
        
        modeling_options['General']['openfast_configuration']['mpi_run'] = False
        modeling_options['General']['openfast_configuration']['cores']   = 1
        modeling_options['General']['openfast_configuration']['OF_run_dir'] \
            = os.path.join(
                modeling_options['General']['openfast_configuration']['OF_run_dir'],
                '{:08d}'.format(self.design_SN)
            )
        analysis_options['general']['folder_output'] = modeling_options['General']['openfast_configuration']['OF_run_dir']
        
        folder_output = analysis_options['general']['folder_output']
        if not os.path.isdir(folder_output):
            os.makedirs(folder_output)
            
        myopt = PoseOptimizationWEIS(turbine_model, modeling_options, analysis_options)
        wt_opt = om.Problem(
            model=WindPark(modeling_options=modeling_options, opt_options=analysis_options)
        )
        wt_opt.setup(derivatives=False)
        wt_opt = yaml2openmdao(wt_opt, modeling_options, turbine_model, analysis_options)
        wt_opt = assign_ROSCO_values(wt_opt, modeling_options, turbine_model['control'])
        wt_opt = myopt.set_initial(wt_opt, turbine_model)
        wt_opt.run_model()
        
        self.result = wt_opt

    def visualize_turbine(self):

        # Load turbine model
        wt = self.turbine_model

        # Read values
        rotor_diameter = wt['assembly']['rotor_diameter']
        hub_diameter = wt['components']['hub']['diameter']
        hub_height = wt['assembly']['hub_height']
        tower_z_grid = wt['components']['tower']['outer_shape_bem']['reference_axis']['z']['grid']
        tower_z_values = wt['components']['tower']['outer_shape_bem']['reference_axis']['z']['values']
        tower_outer_diameter_grid = wt['components']['tower']['outer_shape_bem']['outer_diameter']['grid']
        tower_outer_diameter_values = wt['components']['tower']['outer_shape_bem']['outer_diameter']['values']
        tower_thickness_grid = wt['components']['tower']['internal_structure_2d_fem']['layers'][0]['thickness']['grid']
        tower_thickness_values = wt['components']['tower']['internal_structure_2d_fem']['layers'][0]['thickness']['values']
        monopile_z_values = wt['components']['monopile']['outer_shape_bem']['reference_axis']['z']['values']
        water_depth = wt['environment']['water_depth']

        # Plot turbine shape
        fgs, axs = plt.subplots(1,2)
        
        # Turbine shape
        axs[0].set_aspect('equal')
        # Monopile shape
        axs[0].add_patch(
            Rectangle(
                (-tower_outer_diameter_values[0]/2.0, monopile_z_values[0]),
                tower_outer_diameter_values[0], tower_z_values[0] - monopile_z_values[0],
                edgecolor='r', fill=False
            )
        )
        # Tower node and shape
        axs[0].plot(np.zeros(shape=(len(tower_z_values)), dtype=float), np.array(tower_z_values), 'k.')
        axs[0].plot(np.array(tower_outer_diameter_values)/2.0, np.array(tower_z_values), 'k-')
        axs[0].plot(-np.array(tower_outer_diameter_values)/2.0, np.array(tower_z_values), 'k-')
        axs[0].plot(
            [-np.array(tower_outer_diameter_values[0])/2.0, np.array(tower_outer_diameter_values[0])/2.0],
            [np.array(tower_z_values)[0], np.array(tower_z_values)[0]], 'k-'
        )
        axs[0].plot(
            [-np.array(tower_outer_diameter_values[-1])/2.0, np.array(tower_outer_diameter_values[-1])/2.0],
            [np.array(tower_z_values)[-1], np.array(tower_z_values)[-1]], 'k-'
        )
        # Hub shape
        axs[0].add_patch(
            Circle(
                (0.0, hub_height), radius=hub_diameter/2.0,
                edgecolor='k', fill=False
            )
        )
        # Rotor shape
        axs[0].add_patch(
            Circle(
                (0.0, hub_height), radius=rotor_diameter/2.0,
                edgecolor='k', fill=False
            )
        )
        axs[0].plot([0.0, 0.0], [hub_height + hub_diameter/2.0, hub_height + rotor_diameter/2.0], 'k-')
        axs[0].plot(
            [hub_diameter/2.0*np.sin(np.pi/3.0), rotor_diameter/2.0*np.sin(np.pi/3.0)],
            [hub_height - hub_diameter/2.0*np.cos(np.pi/3.0), hub_height - rotor_diameter/2.0*np.cos(np.pi/3.0)], 'k-'
        )
        axs[0].plot(
            [-hub_diameter/2.0*np.sin(np.pi/3.0), -rotor_diameter/2.0*np.sin(np.pi/3.0)],
            [hub_height - hub_diameter/2.0*np.cos(np.pi/3.0), hub_height - rotor_diameter/2.0*np.cos(np.pi/3.0)], 'k-'
        )
        # Water
        axs[0].add_patch(
            Rectangle(
                (-rotor_diameter/2.0, -water_depth),
                rotor_diameter, water_depth,
                edgecolor=None, fill=True, facecolor='b', alpha=0.5
            )
        )
        # Soil
        axs[0].add_patch(
            Rectangle(
                (-rotor_diameter/2.0, monopile_z_values[0] - 10.0),
                rotor_diameter, -monopile_z_values[0] - water_depth + 10.0,
                edgecolor=None, fill=True, facecolor='y', alpha=0.5
            )
        )
        # Axis formatting
        axs[0].set_xticks([-rotor_diameter/2.0, 0.0, rotor_diameter/2.0])
        axs[0].set_xlabel('y-axis [m]')
        axs[0].set_ylabel('z-axis [m]')
        axs[0].set_anchor('SW')
        
        # Tower design
        axs1_x = axs[1].twiny()
        axs[1].plot(tower_outer_diameter_values, tower_z_values, 'r-o')
        axs[1].set_xlabel('Tower outer diameter [m]', color='r')
        axs[1].set_ylabel('Tower height [m]')
        axs1_x.plot(np.array(tower_thickness_values)*1000.0, tower_z_values, 'b-o')
        axs1_x.set_xlabel('Tower thickness [mm]', color='b')
        axs[1].set_anchor('SW')

        # Show plot
        fgs.tight_layout()
        fgs.show()

if __name__ == '__main__':
    
    COMPUTE = 2 # 0: None, 1: Cost only, 2: Full model
    
    a = turbine_design()
    a.design = {
        'hub_height': 150.0,
        'tower_top_diameter': 6.5,
        'tower_bottom_thickness': 0.041058,
        'tower_top_thickness': 0.020826,
        'rotor_diameter': 240.0
    }
    a.create_turbine()
    a.visualize_turbine()
    
    if COMPUTE == 0:
        pass
    
    elif COMPUTE == 1:
        a.compute_cost_only()
        AEP = a.result.get_val("rotorse.rp.AEP", units="MW*h")[0]
        LCOE = a.result.get_val('financese.lcoe', units='USD/(MW*h)')[0]
        COST = LCOE*AEP
        print('Estimated yearly cost: {:} USD/year'.format(COST))
        print('over design lifetime of: {:} years'.format(a.turbine_model['assembly']['lifetime']))
        
    elif COMPUTE == 2:
        a.compute_full_model()
        AEP = a.result.get_val("rotorse.rp.AEP", units="MW*h")[0]
        LCOE = a.result.get_val('financese.lcoe', units='USD/(MW*h)')[0]
        COST = LCOE*AEP
        print('Estimated yearly cost: {:} USD/year'.format(COST))
        print('over design lifetime of: {:} years'.format(a.turbine_model['assembly']['lifetime']))
        






