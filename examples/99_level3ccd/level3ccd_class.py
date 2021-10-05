import os
import numpy as np
import openmdao.api as om
import weis
import sqlite3
import yaml

from weis.glue_code.gc_LoadInputs           import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_WT_InitModel       import yaml2openmdao
from weis.glue_code.gc_PoseOptimization     import PoseOptimizationWEIS
from weis.glue_code.glue_code               import WindPark
from weis.glue_code.gc_ROSCOInputs          import assign_ROSCO_values
from weis.control.LinearModel               import LinearTurbineModel
from wisdem.glue_code.gc_PoseOptimization   import PoseOptimization as PoseOptimizationWISDEM

from numpy.matlib                           import repmat
from scipy.interpolate                      import interp1d
from scipy.optimize                         import minimize, approx_fprime
from matplotlib                             import pyplot as plt
from matplotlib.patches                     import (Rectangle, Circle)
from copy                                   import deepcopy
from smt.sampling_methods                   import LHS
from smt.surrogate_models                   import (KRG, KPLS, KPLSK)


class turbine_design:

    def __init__(self, **kwargs):

        self.weis_root = os.path.dirname(os.path.dirname(os.path.abspath(weis.__file__)))
        try:
            self.run_path = os.path.dirname(os.path.abspath(__file__))
        except:
            self.run_path = os.path.abspath(os.curdir)

        if 'wt_name' in kwargs.keys():
            wt_name = kwargs['wt_name']
        else:
            wt_name = 'IEA-15-240-RWT.yaml'

        fname_wt_input = os.path.join(self.weis_root, 'WISDEM', 'examples', '02_reference_turbines', wt_name)
        fname_modeling = os.path.join(self.run_path, 'modeling_options.yaml')
        fname_analysis = os.path.join(self.run_path, 'analysis_options.yaml')
        self.wt_ontology = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling, fname_analysis)
        self.wt_opt = None
        
        turbine_model, modeling_options, analysis_options = self.wt_ontology.get_input_data()
        self.turbine_model = turbine_model
        self.modeling_options = modeling_options
        self.analysis_options = analysis_options
        
        self.keep_States = 'All'
        self.remove_States = [
            'ED Variable speed generator DOF (internal DOF index = DOF_GeAz), rad',
        ]
        self.keep_CntrlInpt = [
            'IfW Extended input: horizontal wind speed (steady/uniform wind), m/s',
            'ED Generator torque, Nm',
            'ED Extended input: collective blade-pitch command, rad'
        ]
        self.remove_CntrlInpt = 'None'
        self.keep_Output = [
            'IfW Wind1VelX, (m/s)',     # 0   X component of wind
            'SrvD GenPwr, (kW)',        # 3   Electrical generator power
            'SrvD GenTq, (kN-m)',       # 4   Electrical generator torque
            'ED GenSpeed, (rpm)',       # 9   Angular speed of the high-speed shaft and generator
            'ED RotSpeed, (rpm)',       # 64  Rotor azimuth angular speed
            'ED RotThrust, (kN)',       # 65  Low-speed shaft thrust force = rotor thrust force
            'ED RotTorq, (kN-m)',       # 66  Low-speed shaft torque = rotor torque
            'ED TwrBsFxt, (kN)',        # 220 Tower base fore-aft shear force
            'ED TwrBsFyt, (kN)',        # 221 Tower base side-to-side shear force
            'ED TwrBsFzt, (kN)',        # 222 Tower base axial force
            'ED TwrBsMxt, (kN-m)',      # 223 Tower base roll (or side-to-side) moment
            'ED TwrBsMyt, (kN-m)',      # 224 Tower base pitching (or fore-aft) moment
            'ED TwrBsMzt, (kN-m)',      # 225 Tower base yaw (or torsional) moment
            'ED YawBrFxp, (kN)',        # 226 Tower-top / yaw bearing fore-aft (nonrotating) shear force
            'ED YawBrFyp, (kN)',        # 227 Tower-top / yaw bearing side-to-side (nonrotating) shear force
            'ED YawBrFzp, (kN)',        # 228 Tower-top / yaw bearing axial force
            'ED YawBrMxp, (kN-m)',      # 229 Nonrotating tower-top / yaw bearing roll moment
            'ED YawBrMyp, (kN-m)',      # 230 Nonrotating tower-top / yaw bearing pitch moment
            'ED YawBrMzp, (kN-m)',      # 231 Tower-top / yaw bearing yaw moment
        ]
        self.remove_Output = 'None'

        self.design = dict()
        self.param = dict()
        self.design_SN = 0
        self.result = None
        self.LinearTurbine = None
        self.linear = dict()
        self.cost_per_year = 0.0
        self.design_life_year = 0.0

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
        self._p_tower_yaw_thickness_default = 0.023998
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
        if 'tower_yaw_thickness' in self.param.keys():
            p_tower_yaw_thickness = self.param['tower_yaw_thickness']
        else:
            p_tower_yaw_thickness = self._p_tower_yaw_thickness_default
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
            'tower_yaw_thickness': p_tower_yaw_thickness,
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
            kind='cubic', fill_value='extrapolate'
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
        tower_thickness_new[tower_thickness_new.shape[0]-2] = p_tower_yaw_thickness
        tower_thickness_new[tower_thickness_new.shape[0]-1] = p_tower_yaw_thickness
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
        
        self.modeling_options = modeling_options
        self.analysis_options = analysis_options
        self.result = wt_opt
        LCOE = wt_opt.get_val('financese.lcoe', units='USD/(MW*h)')[0]
        AEP = wt_opt.get_val("rotorse.rp.AEP", units="MW*h")[0]
        self.cost_per_year = LCOE*AEP
        self.design_life_year = self.turbine_model['assembly']['lifetime']
        print('COST PER YEAR = {:} USD/YEAR'.format(self.cost_per_year))
        print('DESIGN LIFE = {:} YEARS'.format(self.design_life_year))


    def compute_full_model(self, OF_run_dir=None):

        if not self.wt_opt:
            self.pose_wt_model(OF_run_dir)
        self.wt_opt.run_model()
        self.save_linear_model()


    def pose_wt_model(self, OF_run_dir=None):
        
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
        if OF_run_dir:
            modeling_options['General']['openfast_configuration']['OF_run_dir'] = OF_run_dir
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

        self.modeling_options = deepcopy(modeling_options)
        self.analysis_options = deepcopy(analysis_options)
        self.turbine_model = deepcopy(turbine_model)
        self.wt_opt = wt_opt


    def save_linear_model(self, FAST_runDirectory=None, lin_case_name=None):
        
        if not FAST_runDirectory:
            FAST_runDirectory = self.wt_opt.model.aeroelastic.FAST_runDirectory
            lin_case_name = self.wt_opt.model.aeroelastic.lin_case_name
        else:
            if not os.path.isdir(FAST_runDirectory):
                raise ValueError('FAST_runDirectory = {:} does not exist'.format(FAST_runDirectory))
            if not lin_case_name:
                lin_case_name = []
                for fname in os.listdir(FAST_runDirectory):
                    if fname.lower().endswith('.lin'):
                        fname_base = os.path.splitext(os.path.splitext(fname)[0])[0]
                        if fname_base not in lin_case_name:
                            lin_case_name.append(fname_base)
                lin_case_name.sort()

        NLinTimes = self.modeling_options['Level2']['linearization']['NLinTimes']
        LinearTurbine = LinearTurbineModel(
            lin_file_dir=FAST_runDirectory,
            lin_file_names=lin_case_name,
            nlin=NLinTimes,
            remove_azimuth=False
        )
        
        linear_in = {
            'A'             : LinearTurbine.A_ops,
            'B'             : LinearTurbine.B_ops,
            'C'             : LinearTurbine.C_ops,
            'D'             : LinearTurbine.D_ops,
            'x_ops'         : LinearTurbine.x_ops,
            'u_ops'         : LinearTurbine.u_ops,
            'y_ops'         : LinearTurbine.y_ops,
            'DescStates'    : LinearTurbine.DescStates,
            'DescCntrlInpt' : LinearTurbine.DescCntrlInpt,
            'DescOutput'    : LinearTurbine.DescOutput,
            'omega_rpm'     : LinearTurbine.omega_rpm,
            'u_h'           : LinearTurbine.u_h,
            'ind_fast_inps' : LinearTurbine.ind_fast_inps,
            'ind_fast_outs' : LinearTurbine.ind_fast_outs,
        }
        linear_out = self.reduce_linear_model(linear_in)
        
        self.LinearTurbine = LinearTurbine
        self.linear = linear_out


    def reduce_linear_model(self, linear):
        
        A = linear['A']
        B = linear['B']
        C = linear['C']
        D = linear['D']
        x_ops = linear['x_ops']
        u_ops = linear['u_ops']
        y_ops = linear['y_ops']
        DescStates = linear['DescStates']
        DescCntrlInpt = linear['DescCntrlInpt']
        DescOutput = linear['DescOutput']
        omega_rpm = linear['omega_rpm']
        u_h = linear['u_h']
        ind_fast_inps = linear['ind_fast_inps']
        ind_fast_outs = linear['ind_fast_outs']
        
        # Process keep_States
        if type(self.keep_States) == str:
            if self.keep_States.lower() != 'all':
                raise ValueError('keep_States should be either list or \'all\'')
            else:
                idx_States = list(range(0, len(DescStates)))
        elif type(self.keep_States) == list:
            if len(self.keep_States) == 0:
                raise ValueError('keep_States list should include at least one item')
            else:
                idx_States = [
                    DescStates.index(s) for s in self.keep_States
                ]
        else:
            raise ValueError('keep_States should be either list or \'all\'')
        
        # Process remove_States
        if type(self.remove_States) == str:
            if self.remove_States.lower() != 'none':
                raise ValueError('remove_States should be either list or \'none\'')
            else:
                idx_remove_States = []
        elif type(self.remove_States) == list:
            idx_remove_States = [
                DescStates.index(s) for s in self.remove_States
            ]
        else:
            raise ValueError('remove_States should be either list or \'none\'')
        idx_States = np.delete(idx_States, idx_remove_States).tolist()
        
        # Process keep_CntrlInpt
        if type(self.keep_CntrlInpt) == str:
            if self.keep_CntrlInpt.lower() != 'all':
                raise ValueError('keep_CntrlInpt should be either list or \'all\'')
            else:
                idx_CntrlInpt = list(range(0, len(DescCntrlInpt)))
        elif type(self.keep_CntrlInpt) == list:
            if len(self.keep_CntrlInpt) == 0:
                raise ValueError('keep_CntrlInpt list should include at least one item')
            else:
                idx_CntrlInpt = [
                    DescCntrlInpt.index(s) for s in self.keep_CntrlInpt
                ]
        else:
            raise ValueError('keep_CntrlInpt should be either list or \'all\'')
        
        # Process remove_CntrlInpt
        if type(self.remove_CntrlInpt) == str:
            if self.remove_CntrlInpt.lower() != 'none':
                raise ValueError('remove_CntrlInpt should be either list or \'none\'')
            else:
                idx_remove_CntrlInpt = []
        elif type(self.remove_CntrlInpt) == list:
            idx_remove_CntrlInpt = [
                DescCntrlInpt.index(s) for s in self.remove_CntrlInpt
            ]
        else:
            raise ValueError('remove_CntrlInpt should be either list or \'none\'')
        idx_CntrlInpt = np.delete(idx_CntrlInpt, idx_remove_CntrlInpt).tolist()
        
        # Process keep_Output
        if type(self.keep_Output) == str:
            if self.keep_Output.lower() != 'all':
                raise ValueError('keep_Output should be either list or \'all\'')
            else:
                idx_Output = list(range(0, len(DescOutput)))
        elif type(self.keep_Output) == list:
            if len(self.keep_Output) == 0:
                raise ValueError('keep_Output list should include at least one item')
            else:
                idx_Output = [
                    DescOutput.index(s) for s in self.keep_Output
                ]
        else:
            raise ValueError('keep_Output should be either list or \'all\'')
        
        # Process remove_Output
        if type(self.remove_Output) == str:
            if self.remove_Output.lower() != 'none':
                raise ValueError('remove_Output should be either list or \'none\'')
            else:
                idx_remove_Output = []
        elif type(self.remove_Output) == list:
            idx_remove_Output = [
                DescOutput.index(s) for s in self.remove_Output
            ]
        else:
            raise ValueError('remove_Output should be either list or \'none\'')
        idx_Output = np.delete(idx_Output, idx_remove_Output).tolist()
        
        Ar = A[idx_States,:,:][:,idx_States,:].tolist()
        Br = B[idx_States,:,:][:,idx_CntrlInpt,:].tolist()
        Cr = C[idx_Output,:,:][:,idx_States,:].tolist()
        Dr = D[idx_Output,:,:][:,idx_CntrlInpt,:].tolist()
        
        x_ops_r = x_ops[idx_States,:].tolist()
        u_ops_r = u_ops[idx_CntrlInpt,:].tolist()
        y_ops_r = y_ops[idx_Output,:].tolist()
        
        DescStates_r = [DescStates[i] for i in idx_States]
        DescCntrlInpt_r = [DescCntrlInpt[i] for i in idx_CntrlInpt]
        DescOutput_r = [DescOutput[i] for i in idx_Output]
        
        omega_rpm_r = np.mean(omega_rpm, axis=0).tolist()
        u_h_r = u_h.tolist()
        
        ind_fast_inps_r = ind_fast_inps[idx_CntrlInpt].tolist()
        ind_fast_outs_r = ind_fast_outs[idx_Output].tolist()
        
        # Return results
        linear_out = {
            'A'             : Ar,
            'B'             : Br,
            'C'             : Cr,
            'D'             : Dr,
            'x_ops'         : x_ops_r,
            'u_ops'         : u_ops_r,
            'y_ops'         : y_ops_r,
            'DescStates'    : DescStates_r,
            'DescCntrlInpt' : DescCntrlInpt_r,
            'DescOutput'    : DescOutput_r,
            'omega_rpm'     : omega_rpm_r,
            'u_h'           : u_h_r,
            'ind_fast_inps' : ind_fast_inps_r,
            'ind_fast_outs' : ind_fast_outs_r,
        }
        return linear_out


    def visualize_turbine(self):

        # Load turbine model
        wt = self.turbine_model

        # Read values
        rotor_diameter = wt['assembly']['rotor_diameter']
        hub_diameter = wt['components']['hub']['diameter']
        hub_height = wt['assembly']['hub_height']
        #tower_z_grid = wt['components']['tower']['outer_shape_bem']['reference_axis']['z']['grid']
        tower_z_values = wt['components']['tower']['outer_shape_bem']['reference_axis']['z']['values']
        #tower_outer_diameter_grid = wt['components']['tower']['outer_shape_bem']['outer_diameter']['grid']
        tower_outer_diameter_values = wt['components']['tower']['outer_shape_bem']['outer_diameter']['values']
        #tower_thickness_grid = wt['components']['tower']['internal_structure_2d_fem']['layers'][0]['thickness']['grid']
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


class sql_design:
    
    def __init__(self, **kwargs):
        
        self.dbpath = ''
        self.conn = None
        self.cursor = None
        self.design_table_name = 'design'
        self.param_list = {
            'tower_div': 'INTEGER',
            'tower_bottom_height': 'REAL',
            'tower_bottom_diameter': 'REAL',
            'tower_yaw_thickness': 'REAL',
            'monopile_bottom_height': 'REAL',
            'water_depth': 'REAL'
        }
        self.design_list = {
            'hub_height': 'REAL',
            'tower_top_diameter': 'REAL',
            'tower_bottom_thickness': 'REAL',
            'tower_top_thickness': 'REAL',
            'rotor_diameter': 'REAL',
        }
        
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass


    def create_connection(self):
        
        if self.dbpath == '':
            raise ValueError('dbpath should be specified.')
        
        dbdirpath = os.path.dirname(self.dbpath)
        if not os.path.isdir(dbdirpath):
            os.makedirs(dbdirpath)
        
        conn = None
        try:
            conn = sqlite3.connect(self.dbpath)
        except sqlite3.Error as e:
            print(e)
            
        self.conn = conn
        if self.conn:
            self.cursor = self.conn.cursor()
        else:
            self.cursor = None


    def remove_db(self):
        if os.path.isfile(self.dbpath):
            try:
                os.remove(self.dbpath)
            except OSError:
                print('DB file cannot be removed.')


    def close_connection(self):
        
        if self.cursor:
            self.cursor.close()
        
        if self.conn:
            self.conn.close()


    def create_table(self):
        
        if self.conn == None:
            raise Exception('DB is not connected.')
            
        if self.cursor == None:
            try:
                self.cursor = self.conn.cursor()
            except sqlite3.Error as e:
                print(e)
                raise Exception('Cursor cannot be created.')
            
        drop_table_sql = 'DROP TABLE IF EXISTS ' + \
            self.design_table_name + ';'
        create_table_sql = 'CREATE TABLE IF NOT EXISTS ' + \
            self.design_table_name + ' ( ' + \
            'id INTEGER NOT NULL PRIMARY KEY'
        for k in self.design_list.keys():
            create_table_sql += ', ' + k + ' ' + self.design_list[k] + '  NOT NULL DEFAULT 0.0'
        for k in self.param_list.keys():
            create_table_sql += ', ' + k + ' ' + self.param_list[k] + '  NOT NULL DEFAULT 0.0'
        create_table_sql += ' );'
        
        # Get cursur, drop existing table, and create new table
        try:
            self.cursor.execute(drop_table_sql)
            self.cursor.execute(create_table_sql)
        except sqlite3.Error as e:
            print(e)


    def add_data(self, design, param):
        
        if self.conn == None:
            self.create_connection()
            self.create_table()
        
        design_list_keys = list(self.design_list.keys())
        param_list_keys = list(self.param_list.keys())
        
        sql = 'INSERT INTO ' + self.design_table_name + '('
        for idx in range(len(design_list_keys)):
            sql += design_list_keys[idx]
            sql += ','
        for idx in range(len(param_list_keys)):
            sql += param_list_keys[idx]
            if idx != len(param_list_keys) - 1:
                sql += ','
        sql += ') VALUES('
        for idx in range(len(design_list_keys)):
            sql += '?'
            sql += ','
        for idx in range(len(param_list_keys)):
            sql += '?'
            if idx != len(param_list_keys) - 1:
                sql += ','
        sql += ');'
        
        val = []
        for idx in range(len(design_list_keys)):
            val.append(design[design_list_keys[idx]])
        for idx in range(len(param_list_keys)):
            val.append(param[param_list_keys[idx]])
            
        try:
            self.cursor.execute(sql, tuple(val))
        except sqlite3.Error as e:
            print(e)


    def get_design_id(self):
        
        if self.conn == None:
            return []
        
        if self.cursor == None:
            try:
                self.cursor = self.conn.cursor()
            except sqlite3.Error as e:
                print(e)
                return []
        
        sql = 'SELECT id FROM ' + self.design_table_name + ' ORDER BY id ASC;'
        
        try:
            self.cursor.execute(sql)
            cout = self.cursor.fetchall()
            out = [i[0] for i in cout]
        except sqlite3.Error as e:
            print(e)
            out = []
            
        return out


    def get_design_dict(self, id_num):
        
        if self.conn == None:
            raise Exception('DB connection lost.')
        
        if self.cursor == None:
            try:
                self.cursor = self.conn.cursor()
            except sqlite3.Error as e:
                print(e)
                raise Exception('DB cursor lost.')
        
        sql = 'SELECT * FROM ' + self.design_table_name + ' WHERE id=' + '{:d}'.format(id_num) + ';'
        
        try:
            self.cursor.execute(sql)
            cout = self.cursor.fetchall()
        except sqlite3.Error as e:
            print(e)
            return None
        
        if len(cout) != 1:
            raise Exception('Number of design for specified id is not 1.')
        
        out = cout[0]
        design_list_keys = list(self.design_list.keys())
        param_list_keys = list(self.param_list.keys())
        
        design = {}
        param = {}
        for idx in range(len(design_list_keys)):
            design[design_list_keys[idx]] = out[idx + 1]
        for idx in range(len(param_list_keys)):
            param[param_list_keys[idx]] = out[len(design_list_keys) + idx + 1]
        
        return design, param


class surrogate_model:
    
    def __init__(self):
        
        self._n_dim_x = 0
        self._n_dim_f = 0
        self._x_train = None
        self._f_train = None
        self.sampling_model = None
        self._x_sampling = None
        self.surrogate_model = None
        self._sm = None


    def add_train_pts(self, x_train, f_train):
        
        x_train = np.array(x_train)
        f_train = np.array(f_train)
        
        if x_train.shape[0] != f_train.shape[0]:
            raise ValueError('number of training points do not match between x and f.')
        
        if self._n_dim_x == 0:
            self._n_dim_x = x_train.shape[1]
        else:
            if self._n_dim_x != x_train.shape[1]:
                raise ValueError('x dimension of training points does not match to the dimension of existing points.')
        
        if self._n_dim_f == 0:
            self._n_dim_f = f_train.shape[1]
        else:
            if self._n_dim_f != f_train.shape[1]:
                raise ValueError('f dimension of training points does not match to the dimension of existing points.')
        
        if self._x_train == None:
            self._x_train = np.zeros(shape=(0, self._n_dim_x), dtype=float)
            
        if self._f_train == None:
            self._f_train = np.zeros(shape=(0, self._n_dim_f), dtype=float)
            
        if self._x_train.shape[0] != self._f_train.shape[0]:
            raise Exception('number of existing points do not match between x and f.')
            
        self._x_train = np.append(
            self._x_train,
            x_train,
            axis = 0
        )
        
        self._f_train = np.append(
            self._f_train,
            f_train,
            axis = 0
        )


    def sampling(self, nt, xlimits, criterion='ese', random_state=0, extreme=True):
        
        dim = xlimits.shape[0]
        
        if (type(self.sampling_model) != LHS) and extreme:
            xe = np.array(
                np.meshgrid(
                    *[xlimits[i,:].tolist() for i in range(dim)]
                )
            ).T.reshape(-1, dim)
            
            ne = xe.shape[0]
            if nt > 3*ne:
                ns = nt - ne
            else:
                ns = int(np.ceil(nt*2.0/3.0))
        else:
            xe = np.zeros(shape=(0, dim), dtype=float)
            ns = nt
        
        if type(self.sampling_model) != LHS:
            if criterion.lower() not in ['center', 'maximin', 'centermaximin', 'correlation', 'c', 'm', 'cm', 'corr', 'ese']:
                self.sampling_model = LHS(
                    xlimits = xlimits,
                    criterion='ese',
                    random_state = random_state
                )
            else:
                self.sampling_model = LHS(
                    xlimits = xlimits,
                    criterion=criterion.lower(),
                    random_state = random_state
                )
            xs = self.sampling_model(ns)
            self._x_sampling = np.append(xe, xs, axis=0)
            return self._x_sampling
        else:
            return self._x_sampling


    def split_list_chunks(self, fulllist, max_n_chunk=1, item_count=None):

        item_count = item_count or len(fulllist)
        n_chunks = min(item_count, max_n_chunk)
        fulllist = iter(fulllist)
        floor = item_count // n_chunks
        ceiling = floor + 1
        stepdown = item_count % n_chunks
        for x_i in range(n_chunks):
            length = ceiling if x_i < stepdown else floor
            yield [next(fulllist) for _ in range(length)]


    def training(self):

        n_dim_x = self._n_dim_x
        n_dim_f = self._n_dim_f
        x_train = np.array(self._x_train)
        f_train = np.array(self._f_train)

        if self.surrogate_model.lower() == 'kpls':
            self._sm = KPLS(print_global=False)
        elif self.surrogate_model.lower() == 'kplsk':
            self._sm = KPLSK(print_global=False)
        else:
            self._sm = KRG(print_global=False)
        self._sm.set_training_values(x_train, f_train)
        self._sm.train()


    def predict(self, x):

        return self._sm.predict_values(x)
        

class dfsm_class:

    def __init__(self):

        self._A_shape = None
        self._A_len = None
        self._B_shape = None
        self._B_len = None
        self._C_shape = None
        self._C_len = None
        self._D_shape = None
        self._D_len = None
        self.P_TRAIN = None
        self.F_TRAIN_A = None
        self.F_TRAIN_B = None
        self.F_TRAIN_C = None
        self.F_TRAIN_D = None
        self.F_TRAIN_W_OPS = None
        self.F_TRAIN_X_OPS = None
        self.F_TRAIN_U_OPS = None
        self.F_TRAIN_Y_OPS = None
        self.F_TRAIN_COST = None
        self.D_TRAIN = None
        self.D_TRAIN_A = None
        self.D_TRAIN_B = None
        self.D_TRAIN_C = None
        self.D_TRAIN_D = None
        self.D_TRAIN_W_OPS = None
        self.D_TRAIN_X_OPS = None
        self.D_TRAIN_U_OPS = None
        self.D_TRAIN_Y_OPS = None
        self.D_TRAIN_COST = None
        self.SM_A = None
        self.SM_B = None
        self.SM_C = None
        self.SM_D = None
        self.SM_W_OPS = None
        self.SM_X_OPS = None
        self.SM_U_OPS = None
        self.SM_Y_OPS = None
        self.SM_COST = None
        self.dbpath = None
        self.surrogate_model = 'KRG'


    def load_linear_models(self, dbpath):

        self.dbpath = dbpath

        d = sql_design(dbpath = dbpath)
        d.create_connection()
        total_design_id_list = d.get_design_id()
        d.close_connection()

        ympath = os.path.dirname(dbpath)

        P_TRAIN = None
        F_TRAIN_A = None
        F_TRAIN_B = None
        F_TRAIN_C = None
        F_TRAIN_D = None
        F_TRAIN_W_OPS = None
        F_TRAIN_X_OPS = None
        F_TRAIN_U_OPS = None
        F_TRAIN_Y_OPS = None
        F_TRAIN_COST = None

        for id_val in total_design_id_list:

            ymfpath = os.path.join(ympath, '{:08d}.yaml'.format(id_val))

            if os.path.isfile(ymfpath):

                with open(ymfpath, 'rt') as yml:
                    dataset = yaml.safe_load(yml)
                
                u_h = np.array(dataset['u_h'])
                hub_height = dataset['design']['hub_height']*np.ones(u_h.shape)
                rotor_diameter = dataset['design']['rotor_diameter']*np.ones(u_h.shape)
                tower_bottom_thickness = dataset['design']['tower_bottom_thickness']*np.ones(u_h.shape)
                tower_top_diameter = dataset['design']['tower_top_diameter']*np.ones(u_h.shape)
                tower_top_thickness = dataset['design']['tower_top_thickness']*np.ones(u_h.shape)
                
                p_train = np.append(
                    np.array([u_h]).transpose(),
                    np.append(
                        np.array([hub_height]).transpose(),
                        np.append(
                            np.array([tower_top_diameter]).transpose(),
                            np.append(
                                np.array([tower_bottom_thickness]).transpose(),
                                np.append(
                                    np.array([tower_top_thickness]).transpose(),
                                    np.array([rotor_diameter]).transpose(),
                                    axis=1
                                ),
                                axis=1
                            ),
                            axis=1
                        ),
                        axis=1
                    ),
                    axis=1
                )
                if type(P_TRAIN) == type(None):
                    P_TRAIN = deepcopy(p_train)
                else:
                    P_TRAIN = deepcopy(np.append(P_TRAIN, p_train, axis=0))

                A_3d = np.array(dataset['A'])
                B_3d = np.array(dataset['B'])
                C_3d = np.array(dataset['C'])
                D_3d = np.array(dataset['D'])
                if type(self._A_shape) == type(None):
                    self._A_shape = list(A_3d.shape)
                    self._A_len = self._A_shape[0]*self._A_shape[1]
                else:
                    if not list(A_3d.shape) == self._A_shape:
                        raise ValueError('Matrix A dimensions are not consistent.')
                    if not self._A_shape[2] == len(u_h):
                        raise ValueError('Missing wind speeds in matrix A.')
                if type(self._B_shape) == type(None):
                    self._B_shape = list(B_3d.shape)
                    self._B_len = self._B_shape[0]*self._B_shape[1]
                else:
                    if not list(B_3d.shape) == self._B_shape:
                        raise ValueError('Matrix B dimensions are not consistent.')
                    if not self._B_shape[2] == len(u_h):
                        raise ValueError('Missing wind speeds in matrix B.')
                if type(self._C_shape) == type(None):
                    self._C_shape = list(C_3d.shape)
                    self._C_len = self._C_shape[0]*self._C_shape[1]
                else:
                    if not list(C_3d.shape) == self._C_shape:
                        raise ValueError('Matrix C dimensions are not consistent.')
                    if not self._C_shape[2] == len(u_h):
                        raise ValueError('Missing wind speeds in matrix C.')
                if type(self._D_shape) == type(None):
                    self._D_shape = list(D_3d.shape)
                    self._D_len = self._D_shape[0]*self._D_shape[1]
                else:
                    if not list(D_3d.shape) == self._D_shape:
                        raise ValueError('Matrix D dimensions are not consistent.')
                    if not self._D_shape[2] == len(u_h):
                        raise ValueError('Missing wind speeds in matrix D.')
                
                f_train_A = np.zeros((0, self._A_len), dtype=float)
                f_train_B = np.zeros((0, self._B_len), dtype=float)
                f_train_C = np.zeros((0, self._C_len), dtype=float)
                f_train_D = np.zeros((0, self._D_len), dtype=float)
                for idx in range(0, len(u_h)):
                    f_train_A = np.append(
                        f_train_A,
                        np.array([A_3d[:,:,idx].flatten()]),
                        axis=0
                    )
                    f_train_B = np.append(
                        f_train_B,
                        np.array([B_3d[:,:,idx].flatten()]),
                        axis=0
                    )
                    f_train_C = np.append(
                        f_train_C,
                        np.array([C_3d[:,:,idx].flatten()]),
                        axis=0
                    )
                    f_train_D = np.append(
                        f_train_D,
                        np.array([D_3d[:,:,idx].flatten()]),
                        axis=0
                    )
                
                if type(F_TRAIN_A) == type(None):
                    F_TRAIN_A = deepcopy(f_train_A)
                else:
                    F_TRAIN_A = deepcopy(np.append(F_TRAIN_A, f_train_A, axis=0))
                if type(F_TRAIN_B) == type(None):
                    F_TRAIN_B = deepcopy(f_train_B)
                else:
                    F_TRAIN_B = deepcopy(np.append(F_TRAIN_B, f_train_B, axis=0))
                if type(F_TRAIN_C) == type(None):
                    F_TRAIN_C = deepcopy(f_train_C)
                else:
                    F_TRAIN_C = deepcopy(np.append(F_TRAIN_C, f_train_C, axis=0))
                if type(F_TRAIN_D) == type(None):
                    F_TRAIN_D = deepcopy(f_train_D)
                else:
                    F_TRAIN_D = deepcopy(np.append(F_TRAIN_D, f_train_D, axis=0))
                
                f_train_w_ops = np.array([dataset['omega_rpm']]).transpose()
                f_train_x_ops = np.array(dataset['x_ops']).transpose()
                f_train_u_ops = np.array(dataset['u_ops']).transpose()
                f_train_y_ops = np.array(dataset['y_ops']).transpose()
                f_train_cost = dataset['cost_per_year']*np.ones(f_train_w_ops.shape, dtype=float)

                if type(F_TRAIN_W_OPS) == type(None):
                    F_TRAIN_W_OPS = deepcopy(f_train_w_ops)
                else:
                    F_TRAIN_W_OPS = deepcopy(np.append(F_TRAIN_W_OPS, f_train_w_ops, axis=0))
                if type(F_TRAIN_X_OPS) == type(None):
                    F_TRAIN_X_OPS = deepcopy(f_train_x_ops)
                else:
                    F_TRAIN_X_OPS = deepcopy(np.append(F_TRAIN_X_OPS, f_train_x_ops, axis=0))
                if type(F_TRAIN_U_OPS) == type(None):
                    F_TRAIN_U_OPS = deepcopy(f_train_u_ops)
                else:
                    F_TRAIN_U_OPS = deepcopy(np.append(F_TRAIN_U_OPS, f_train_u_ops, axis=0))
                if type(F_TRAIN_Y_OPS) == type(None):
                    F_TRAIN_Y_OPS = deepcopy(f_train_y_ops)
                else:
                    F_TRAIN_Y_OPS = deepcopy(np.append(F_TRAIN_Y_OPS, f_train_y_ops, axis=0))
                if type(F_TRAIN_COST) == type(None):
                    F_TRAIN_COST = deepcopy(f_train_cost)
                else:
                    F_TRAIN_COST = deepcopy(np.append(F_TRAIN_COST, f_train_cost, axis=0))

        self.P_TRAIN = P_TRAIN
        self.F_TRAIN_A = F_TRAIN_A
        self.F_TRAIN_B = F_TRAIN_B
        self.F_TRAIN_C = F_TRAIN_C
        self.F_TRAIN_D = F_TRAIN_D
        self.F_TRAIN_W_OPS = F_TRAIN_W_OPS
        self.F_TRAIN_X_OPS = F_TRAIN_X_OPS
        self.F_TRAIN_U_OPS = F_TRAIN_U_OPS
        self.F_TRAIN_Y_OPS = F_TRAIN_Y_OPS
        self.F_TRAIN_COST = F_TRAIN_COST
        self._A_shape[2] = self.F_TRAIN_A.shape[0]
        self._B_shape[2] = self.F_TRAIN_B.shape[0]
        self._C_shape[2] = self.F_TRAIN_C.shape[0]
        self._D_shape[2] = self.F_TRAIN_D.shape[0]


    def train_sm(self):

        print('Scaling data for DFSM training...')
        self.D_TRAIN = np.max(self.P_TRAIN, axis=0) - np.min(self.P_TRAIN, axis=0)
        self.D_TRAIN = np.where(self.D_TRAIN < 1.0e-9, 1.0, self.D_TRAIN)
        P_TRAIN = self.P_TRAIN/repmat(self.D_TRAIN, self.P_TRAIN.shape[0], 1)

        self.D_TRAIN_A = np.max(self.F_TRAIN_A, axis=0) - np.min(self.F_TRAIN_A, axis=0)
        self.D_TRAIN_A = np.where(self.D_TRAIN_A < 1.0e-9, 1.0, self.D_TRAIN_A)
        F_TRAIN_A = self.F_TRAIN_A/repmat(self.D_TRAIN_A, self.F_TRAIN_A.shape[0], 1)

        self.D_TRAIN_B = np.max(self.F_TRAIN_B, axis=0) - np.min(self.F_TRAIN_B, axis=0)
        self.D_TRAIN_B = np.where(self.D_TRAIN_B < 1.0e-9, 1.0, self.D_TRAIN_B)
        F_TRAIN_B = self.F_TRAIN_B/repmat(self.D_TRAIN_B, self.F_TRAIN_B.shape[0], 1)

        self.D_TRAIN_C = np.max(self.F_TRAIN_C, axis=0) - np.min(self.F_TRAIN_C, axis=0)
        self.D_TRAIN_C = np.where(self.D_TRAIN_C < 1.0e-9, 1.0, self.D_TRAIN_C)
        F_TRAIN_C = self.F_TRAIN_C/repmat(self.D_TRAIN_C, self.F_TRAIN_C.shape[0], 1)

        self.D_TRAIN_D = np.max(self.F_TRAIN_D, axis=0) - np.min(self.F_TRAIN_D, axis=0)
        self.D_TRAIN_D = np.where(self.D_TRAIN_D < 1.0e-9, 1.0, self.D_TRAIN_D)
        F_TRAIN_D = self.F_TRAIN_D/repmat(self.D_TRAIN_D, self.F_TRAIN_D.shape[0], 1)

        self.D_TRAIN_W_OPS = np.max(self.F_TRAIN_W_OPS, axis=0) - np.min(self.F_TRAIN_W_OPS, axis=0)
        self.D_TRAIN_W_OPS = np.where(self.D_TRAIN_W_OPS < 1.0e-9, 1.0, self.D_TRAIN_W_OPS)
        F_TRAIN_W_OPS = self.F_TRAIN_W_OPS/repmat(self.D_TRAIN_W_OPS, self.F_TRAIN_W_OPS.shape[0], 1)

        self.D_TRAIN_X_OPS = np.max(self.F_TRAIN_X_OPS, axis=0) - np.min(self.F_TRAIN_X_OPS, axis=0)
        self.D_TRAIN_X_OPS = np.where(self.D_TRAIN_X_OPS < 1.0e-9, 1.0, self.D_TRAIN_X_OPS)
        F_TRAIN_X_OPS = self.F_TRAIN_X_OPS/repmat(self.D_TRAIN_X_OPS, self.F_TRAIN_X_OPS.shape[0], 1)
        
        self.D_TRAIN_U_OPS = np.max(self.F_TRAIN_U_OPS, axis=0) - np.min(self.F_TRAIN_U_OPS, axis=0)
        self.D_TRAIN_U_OPS = np.where(self.D_TRAIN_U_OPS < 1.0e-9, 1.0, self.D_TRAIN_U_OPS)
        F_TRAIN_U_OPS = self.F_TRAIN_U_OPS/repmat(self.D_TRAIN_U_OPS, self.F_TRAIN_U_OPS.shape[0], 1)

        self.D_TRAIN_Y_OPS = np.max(self.F_TRAIN_Y_OPS, axis=0) - np.min(self.F_TRAIN_Y_OPS, axis=0)
        self.D_TRAIN_Y_OPS = np.where(self.D_TRAIN_Y_OPS < 1.0e-9, 1.0, self.D_TRAIN_Y_OPS)
        F_TRAIN_Y_OPS = self.F_TRAIN_Y_OPS/repmat(self.D_TRAIN_Y_OPS, self.F_TRAIN_Y_OPS.shape[0], 1)

        self.D_TRAIN_COST = np.max(self.F_TRAIN_COST, axis=0) - np.min(self.F_TRAIN_COST, axis=0)
        self.D_TRAIN_COST = np.where(self.D_TRAIN_COST < 1.0e-9, 1.0, self.D_TRAIN_COST)
        F_TRAIN_COST = self.F_TRAIN_COST/repmat(self.D_TRAIN_COST, self.F_TRAIN_COST.shape[0], 1)

        print('Training surrogate model for A matrix...')
        self.SM_A = surrogate_model()
        self.SM_A.surrogate_model = self.surrogate_model
        self.SM_A.add_train_pts(P_TRAIN, F_TRAIN_A)
        self.SM_A.training()
    
        print('Training surrogate model for B matrix...')
        self.SM_B = surrogate_model()
        self.SM_B.surrogate_model = self.surrogate_model
        self.SM_B.add_train_pts(P_TRAIN, F_TRAIN_B)
        self.SM_B.training()
    
        print('Training surrogate model for C matrix...')
        self.SM_C = surrogate_model()
        self.SM_C.surrogate_model = self.surrogate_model
        self.SM_C.add_train_pts(P_TRAIN, F_TRAIN_C)
        self.SM_C.training()
    
        print('Training surrogate model for D matrix...')
        self.SM_D = surrogate_model()
        self.SM_D.surrogate_model = self.surrogate_model
        self.SM_D.add_train_pts(P_TRAIN, F_TRAIN_D)
        self.SM_D.training()
    
        print('Training surrogate model for operating point for Omega...')
        self.SM_W_OPS = surrogate_model()
        self.SM_W_OPS.surrogate_model = self.surrogate_model
        self.SM_W_OPS.add_train_pts(P_TRAIN, F_TRAIN_W_OPS)
        self.SM_W_OPS.training()
    
        print('Training surrogate model for operating point for States...')
        self.SM_X_OPS = surrogate_model()
        self.SM_X_OPS.surrogate_model = self.surrogate_model
        self.SM_X_OPS.add_train_pts(P_TRAIN, F_TRAIN_X_OPS)
        self.SM_X_OPS.training()
    
        print('Training surrogate model for operating point for Inputs...')
        self.SM_U_OPS = surrogate_model()
        self.SM_U_OPS.surrogate_model = self.surrogate_model
        self.SM_U_OPS.add_train_pts(P_TRAIN, F_TRAIN_U_OPS)
        self.SM_U_OPS.training()
    
        print('Training surrogate model for operating point for Outputs...')
        self.SM_Y_OPS = surrogate_model()
        self.SM_Y_OPS.surrogate_model = self.surrogate_model
        self.SM_Y_OPS.add_train_pts(P_TRAIN, F_TRAIN_Y_OPS)
        self.SM_Y_OPS.training()

        print('Training surrogate model for operating point for Cost...')
        self.SM_COST = surrogate_model()
        self.SM_COST.surrogate_model = self.surrogate_model
        self.SM_COST.add_train_pts(P_TRAIN, F_TRAIN_COST)
        self.SM_COST.training()

        print('Training completed.')


    def predict_sm_A(self, p, squeeze=True):

        pD = repmat(self.D_TRAIN, p.shape[0], 1)
        sD = repmat(self.D_TRAIN_A, p.shape[0], 1)
        f = self.SM_A.predict(p/pD)*sD
        nt = f.shape[0]
        nd1, nd2 = self._A_shape[0:2]
        if squeeze:
            return np.squeeze(np.moveaxis(f.reshape(nt, nd1, nd2), 0, -1))
        else:
            return np.moveaxis(f.reshape(nt, nd1, nd2), 0, -1)


    def predict_sm_B(self, p, squeeze=True):

        pD = repmat(self.D_TRAIN, p.shape[0], 1)
        sD = repmat(self.D_TRAIN_B, p.shape[0], 1)
        f = self.SM_B.predict(p/pD)*sD
        nt = f.shape[0]
        nd1, nd2 = self._B_shape[0:2]
        if squeeze:
            return np.squeeze(np.moveaxis(f.reshape(nt, nd1, nd2), 0, -1))
        else:
            return np.moveaxis(f.reshape(nt, nd1, nd2), 0, -1)


    def predict_sm_C(self, p, squeeze=True):

        pD = repmat(self.D_TRAIN, p.shape[0], 1)
        sD = repmat(self.D_TRAIN_C, p.shape[0], 1)
        f = self.SM_C.predict(p/pD)*sD
        nt = f.shape[0]
        nd1, nd2 = self._C_shape[0:2]
        if squeeze:
            return np.squeeze(np.moveaxis(f.reshape(nt, nd1, nd2), 0, -1))
        else:
            return np.moveaxis(f.reshape(nt, nd1, nd2), 0, -1)


    def predict_sm_D(self, p, squeeze=True):

        pD = repmat(self.D_TRAIN, p.shape[0], 1)
        sD = repmat(self.D_TRAIN_D, p.shape[0], 1)
        f = self.SM_D.predict(p/pD)*sD
        nt = f.shape[0]
        nd1, nd2 = self._D_shape[0:2]
        if squeeze:
            return np.squeeze(np.moveaxis(f.reshape(nt, nd1, nd2), 0, -1))
        else:
            return np.moveaxis(f.reshape(nt, nd1, nd2), 0, -1)


    def predict_sm_W_OPS(self, p, squeeze=True):

        pD = repmat(self.D_TRAIN, p.shape[0], 1)
        sD = repmat(self.D_TRAIN_W_OPS, p.shape[0], 1)
        f = self.SM_W_OPS.predict(p/pD)*sD
        if squeeze:
            return np.squeeze(f)
        else:
            return f


    def predict_sm_X_OPS(self, p, squeeze=True):

        pD = repmat(self.D_TRAIN, p.shape[0], 1)
        sD = repmat(self.D_TRAIN_X_OPS, p.shape[0], 1)
        f = self.SM_X_OPS.predict(p/pD)*sD
        if squeeze:
            return np.squeeze(f)
        else:
            return f


    def predict_sm_U_OPS(self, p, squeeze=True):

        pD = repmat(self.D_TRAIN, p.shape[0], 1)
        sD = repmat(self.D_TRAIN_U_OPS, p.shape[0], 1)
        f = self.SM_U_OPS.predict(p/pD)*sD
        if squeeze:
            return np.squeeze(f)
        else:
            return f


    def predict_sm_Y_OPS(self, p, squeeze=True):

        pD = repmat(self.D_TRAIN, p.shape[0], 1)
        sD = repmat(self.D_TRAIN_Y_OPS, p.shape[0], 1)
        f = self.SM_Y_OPS.predict(p/pD)*sD
        if squeeze:
            return np.squeeze(f)
        else:
            return f
    

    def predict_sm_COST(self, p, squeeze=True):

        pD = repmat(self.D_TRAIN, p.shape[0], 1)
        sD = repmat(self.D_TRAIN_COST, p.shape[0], 1)
        f = self.SM_COST.predict(p/pD)*sD
        if squeeze:
            return np.squeeze(f)
        else:
            return f



class optimization_problem:

    def __init__(self):
        self.num_diff_eps = 1.0e-4
        self.constraint_function_list = []

    def objective(self, x):
        return 0.0

    def gradient(self, x):
        return approx_fprime(x, self.objective, self.num_diff_eps)

    def constraints(self, x):
        if len(self.constraint_function_list) == 0:
            return np.array([], dtype=float)
        else:
            out = []
            for fn in self.constraint_function_list:
                out.append(np.array(fn(x)))

    def jacobian(self, x):
        if len(self.constraint_function_list) == 0:
            return np.array([], dtype=float)
        else:
            out = []
            for fn in self.constraint_function_list:
                out.append(approx_fprime(x, fn, self.num_diff_eps))
            return np.concatenate(out)

    def hessian(self, x):
        h = self.num_diff_eps
        n = x.shape[0]
        fzero = approx_fprime(x, self.objective, self.num_diff_eps)
        row, col = np.nonzero(np.tril(np.ones(shape=(n, n))))
        hess = np.zeros(shape=(n,n), dtype=float)
        for idx in range(n):
            xplus = deepcopy(x)
            xplus[idx] += h
            fplus = approx_fprime(xplus, self.objective, self.num_diff_eps)
            hess[idx, :] = (fplus - fzero)/h
        return hess[row, col]


if __name__ == '__main__':
    
    COMPUTE = 0 # 0: None, 1: Cost only, 2: Full model
    
    # Create DB for design point management
    d = sql_design(dbpath = 'temp/linear_data.db')
    d.remove_db()
    
    # Add DP#1
    design = {
        'hub_height': 150.0,
        'tower_top_diameter': 6.5,
        'tower_bottom_thickness': 0.041058,
        'tower_top_thickness': 0.020826,
        'rotor_diameter': 240.0
    }
    param = {
        'tower_div': 11,
        'tower_bottom_height': 15.0,
        'tower_bottom_diameter': 10.0,
        'tower_yaw_thickness': 0.023998,
        'monopile_bottom_height': -75.0,
        'water_depth': 30.0
    }
    d.add_data(design, param)
    
    # Add DP#2
    design = {
        'hub_height': 155.0,
        'tower_top_diameter': 6.5,
        'tower_bottom_thickness': 0.041058,
        'tower_top_thickness': 0.020826,
        'rotor_diameter': 250.0
    }
    param = {
        'tower_div': 11,
        'tower_bottom_height': 15.0,
        'tower_bottom_diameter': 10.0,
        'tower_yaw_thickness': 0.023998,
        'monopile_bottom_height': -75.0,
        'water_depth': 30.0
    }
    d.add_data(design, param)
    
    # Commit to DB
    d.conn.commit()
    
    # Get the list of DPs in the DB
    id_list = d.get_design_id()
    
    # Evaluate DPs
    for idx in id_list:
        des, par = d.get_design_dict(idx)
        
        a = turbine_design()
        a.design_SN = idx
        a.design = des
        a.param = par
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
            
    d.cursor.close()
    d.conn.close()
