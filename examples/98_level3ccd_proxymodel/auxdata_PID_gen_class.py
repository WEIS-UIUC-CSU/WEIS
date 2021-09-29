#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 14:04:39 2021

@author: bayat2
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import fsolve
from max_tension_function import max_tension
from sympy import integrate
from sympy import Symbol
from sympy.solvers import solve
import sympy as sp
from Spar_Update_params_func import Spar_Update_params_func
import scipy.io
from scipy.interpolate import RectBivariateSpline
from matplotlib.pyplot import plot
from scipy.interpolate import CubicSpline
import scipy
from matplotlib.pyplot import plot, pause

class auxdata:
    def __init__(self,x,wind_avg,wind_type):
        #tower's base outer diameter == platform's tip outer diamet
        d2=6.5;   self.d2=d2           
        
        #parameters defined to calculate tower mass and inertia (page: 176 of the following thesis)
        #(https://escholarship.mcgill.ca/catalog?utf8=%E2%9C%93&search_field=all_fields&q=Dynamics+Modeling%2C+Simulation+and+Analysis+of+a+Floating+Offshore+Wind+Turbine)
        # t_tw(z)=a1 - a2 * z 
        # Do(z)  =a3 - a4 * z
        # z: varibale of tower length. z=0 at its base, and z=l at its tip
        self.a1=x[0,0]
        self.a2=(x[0,0]-x[1,0])/x[3,0]
        self.a3=d2
        self.a4=(d2-x[2,0])/x[3,0]
        self.l=l=x[3,0]
        
        
        z0=np.array([0])
        lb=0
        ub=self.l
        bound_max_tension=Bounds(lb, ub)
        f= lambda z: -max_tension(z,self)
        res = minimize(fun=f,x0=z0,bounds=bound_max_tension)
        self.max_ten_z_buttom=res.x[0]
        
        t_tw_max_ten=self.a1-self.a2*self.max_ten_z_buttom        #tower thickness at maximum stress
        self.Do_max_ten=self.a3-self.a4*self.max_ten_z_buttom     #tower outer diameter at maximum stress
        self.Di_max_ten=self.Do_max_ten-2*(t_tw_max_ten)          #tower inner diameter at maximum stress
        
        # Tower mass and inertia calculation
        # tower length symbolic parameter (to compute tower mass and inertia)
        # z=0 at its base, and z=l at its tip
        z=Symbol('z')
        
        t_tw=self.a1-self.a2*z    #thickness of tower as a function of z
        D_o=self.a3-self.a4*z     #outer diameter of tower as a function of z
        D_i=D_o-2*t_tw  #inner diameter of tower as a function of z

        rho_t=8500.0 ; self.rho_t=rho_t            #tower material density [kg/m^3]  
        
        At=np.pi/4*(D_o**2-D_i**2)  #tower cross section area
        mt_bar=rho_t*At             #mass per length density of tower [kg/m]

        Itxx=np.pi/64*(D_o**4-D_i**4) #second moment of area of tower
        Ityy=np.pi/64*(D_o**4-D_i**4) #second moment of area of tower
        Jt=np.pi/32*(D_o**4-D_i**4)   #second moment of area of tower

        self.mt_bar=mt_bar
        self.mt=np.float64(integrate(mt_bar,(z,0,self.l))) #tower mass
        
        self.z_tw_bar=np.float64(1/self.mt*integrate(z*mt_bar,(z,0,self.l))) #tower center of gravity
        
        Itxb_integrand=(rho_t*Itxx+mt_bar*z**2)*10**(-6)
        Ityb_integrand=(rho_t*Ityy+mt_bar*z**2)*10**(-6)
        self.Itxb=np.float64(10**(6)*integrate(Itxb_integrand,(z,0,self.l))) #tower mass moment of inertia around its base
        self.Ityb=np.float64(10**(6)*integrate(Ityb_integrand,(z,0,self.l))) #tower mass moment of inertia around its base

        Itx=self.Itxb-self.mt*self.z_tw_bar**2 # tower mass moment of inertia around its center of gravity
        Ity=self.Ityb-self.mt*self.z_tw_bar**2 # tower mass moment of inertia around its center of gravity
        self.Itz=np.float64(10**(6)*integrate(rho_t*Jt*10**(-6),(z,0,self.l))) #tower mass moment of inertia around its base
        del(z)
        
        self.g=9.81               # gravity
        mp_baseline=7466330;  # platform baseline mass:
            
        #nacelle mass and inertia:
        mnc=240000;  self.mnc=mnc
        Incx=4901094; 
        Incy=22785; 
        Incz=2607890 ; 
        
        #rotor mass and inertia:
        mr=110000; self.mr=mr
        self.Irx =38759228;   #J_rotor
        Iry=19379614; 
        Irz=19379614 ;
        
        # pltform update (if we change tower parameters, we need to update platform to satisfy equilibrium in heave direction)
        V0_mooring=3*(5.358227171305140e+05)          #baseline mooring
        Fh0_hydrostatic=8.171825533310140e+07         #baseline hydrostatic
        m_T_0=mp_baseline+self.mt+mnc+mr                   #baseline total mass
        delta_F_h0=m_T_0*self.g+V0_mooring-Fh0_hydrostatic #needed additional force in heave direction
        
        R_wall=9.4/2-0.05 #inner radius of platfrom
        delta_l_vb=delta_F_h0/(np.pi*R_wall**2*(1.691729222765398e+03)*self.g) #needed removed length of variable ballast 
        lvb_0=45.098443555583890 #baseline variable ballast length
        lvb=lvb_0-delta_l_vb     #updated variable ballast length
        [mp,zp,Ipx,Ipy,Ipz]=Spar_Update_params_func(lvb) #updated platfrom data
        
        self.nPoints=500 # Points used for numerical quadrature. To get hydrodynamic force,...  we need to calculate some integrals that
                         # are a function of tower length variable (z). ``nPoints'' shows number of
                         # points that are going to be used to calculate those integrals.
       
        """
        Scaling states , controls and their bounds
        states =[v_x v_z \omega_y \Omgea XX ZZ \theta \beta u_{gen}]
        v_x           : platform horizontal velocity in platform body coordinates
        v_z           : platform vertical velocity in platform body coordinates
        \omega_y      : platform rotational velocity in platform body coordinates
        \Omgea        : Rotor rotational velocity in Rotor(==platfrom) body coordinates
        XX           : platform surge in inertial frame
        ZZ           : platform heave in inertial frame
        theta         : platform rotation [pitch] in inertial frame
        \beta         : balde's Pitch angle in Rotor body coordinates
        u_{gen}       : generator torque in Rotor body coordinates
        
        Controls=[\dot{u_{gen}} \dot{\beta}]
        \dot{u_{gen}}      : generator torque rate in Rotor body coordinates
        \dot{\beta}        : pitch angle rate in Rotor body coordinates

        Scaling all states and control signals so that they will be always in [-1,1]
        x:states  , x_{scaled}: scaled states 
        u: control  , u_{scaled}: scaled controls
        x=A+B*x_{scaled} 
        \dot{x_{scaled}}= B^-1*x
        u=C+D*u_{scaled} 

        A(i)=(max(state(i))+min(state(i)))/2
        B(i,i)=(max(state(i))-min(state(i)))/2
        C(i)=(max(control(i))+min(control(i)))/2
        D(i,i)=(max(control(i))-min(control(i)))/2
        """
        self.omega_max=1.5114;  #rad/s 
        self.A_scaled=np.array([[0],[0],[0],[self.omega_max/2],[0],[0],[0],[self.omega_max/2],[0.3403]])
        self.B_scaled=np.diag([3,3,0.1745,self.omega_max/2,100,100,np.deg2rad(6.3),self.omega_max/2,0.3403])
        self.C_scaled=np.array([[0.5*4.18e6],[0.6807/2]])
        self.D_scaled=np.diag([0.5*4.18e6,0.6807/2])

        self.etha=0.944; #drive train efficiency
        self.c_elec=(1-self.etha)*self.A_scaled[3,0]*2/(self.A_scaled[-1,0]*2)
        
        """
        Additional experimental parameters
        hydrodynamic additional damping for [surge, suave, heave, roll, pitch, yaw]
        this additional damping is based on experiment and is reported on the
        following document fo this of platform: https://www.nrel.gov/docs/fy10osti/47535.pdf
        B_addition=np.diag([1e5,1e5,13e4,0,0,13e6])
        
        yaw mooring additional stiffness
        this is reported in page 134
        for this reduced order model we are not using this paraemter because we
        have no yaw motion. our platfrom motion has reduced just to surge, heave,
        and pitch.
        k_add=98.34e6
        """
        self.B_addition=np.diag([1e5,1e5,13e4,0,0,13e6])
        
        #General Parameters
        self.rho_w=1025;      #kg/m^3 (Sea water density)
        self.rho_air= 1.225;  #air density [kg/m^3]
        self.at=130-zp;       #distance between tower base and platform CG, as shown in Figure 2.3. in page: 45

        """ Platform, nacelle, and rotor Parameters"""
        # total geometry (these geometries are difiend in page 38 , and 45
        self.dnc=1.90
        self.dr=5.462 
        self.dnB=2.4                  #distance between tower's tip and rotor axis
        self.Dt=self.at+self.z_tw_bar
        self.Dr=self.at+self.l+self.dnB

        I_Tx=Ipx+Itx+Incx+self.Irx  #system total inertia
        I_Ty=Ipy+Ity+Incy+Iry; #system total inertia
        I_Tz=Ipz+self.Itz+Incz+Irz; #system total inertia
        self.mT=mp+self.mt+mnc+mr;       #system total mass

        #Get Mass matrix based on equations on page 42
        self.M15=self.Dr*(mr+mnc)+self.mt*self.Dt;
        self.M26=self.dnc*mnc-self.dr*mr;
        self.M44=I_Tx+self.Dr**2*(mr+mnc)+self.Dt**2*self.mt;
        self.M55=I_Ty+mr*(self.Dr**2+self.dr**3)+mnc*(self.Dr**2+self.dnc**2)+self.Dt**2*self.mt;
        self.M46=self.Dr*self.dr*mr-self.Dr*self.dnc*mnc;
        self.M66=I_Tz+mr*self.dr**2+mnc*self.dnc**2;

        #mass matrix for reduced system
        self.M_sys=np.array([[self.mT,0,self.M15,0],\
                        [0,self.mT,-self.M26,0],\
                        [self.M15,-self.M26,self.M55,0],\
                        [0,0,0,self.Irx]])
            
        #Hydrostatic
        self.Lc=120;        # Length of platform under water :Fig 2.3
        self.rb=zp;         # Distnace between platform CG and pltform base
        self.aa1=108;       # a faction of Lc defined in Figure 2.8
        self.aa2=14;        # a faction of Lc defined in Figure 2.8
        self.aa3=8;         # a faction of Lc defined in Figure 2.8
        self.d1=9.4;        # Platform base diameter
        self.dtip=self.a3-self.a4*self.l;  # tower tip damater
        
        #HydroDynamics
        self.Ca=0.969954;     # added mass coefficient (also used in added mass)
        self.CD=0.6*1;        # drag coefficient 
        self.Hw=0;            # initial wave amplitude       
        self.Nw=1;
        Tp=200;
        self.Hs=6;
        self.Ha_vec=self.Hs
        self.dw=320;
        self.beta_w=np.deg2rad(0);
        self.rand_vec=0.5;
        self.ww=2*np.pi/Tp;
        hydro_eq=lambda x: self.ww**2-self.g*x*np.tanh(self.dw*x)
        c_aarray=fsolve(hydro_eq,-1e-5)
        self.c=c_aarray[0]
        
        #Mooring Lines (Slack) 
        L0=902.2        #each mooring line length
        w=698.094       #each mooring line weight per unit length
        self.W=w*L0;    #each mooring line weight
        self.r_AGs_1=np.array([[5.2],[-70]])+np.array([[0],[89.915]])      #fairlead-1 position in platform bocy coordinate
        self.r_AGs_2=np.array([[-5.2],[-70]])+np.array([[0],[89.915]])      #fairlead-1 position in platform bocy coordinate
        self.r_EO_1=np.array([[853.87],[-320]])+np.array([[0],[89.915]])    #Anchor-1 position in Inertial frame
        self.r_EO_2=np.array([[-853.87],[-320]])+np.array([[0],[89.915]])   #Anchor-1 position in Inertial frame      
         
        #Load CP and CT
        CP_CT_MAT = scipy.io.loadmat('CP_CT_MAT.mat')
        CP_Mat=CP_CT_MAT['CP_Mat']
        CT_Mat=CP_CT_MAT['CT_Mat']
        Lambda=CP_CT_MAT['Lambda']
        Theta_p=CP_CT_MAT['Theta_p']
        self.THETA_P,self.LAMBDA = np.meshgrid(Theta_p,Lambda)
        self.Cpfunc=RectBivariateSpline(Lambda, Theta_p, CP_Mat, kx=3, ky=3, s=0)
        self.Ctfunc=RectBivariateSpline(Lambda, Theta_p, CT_Mat, kx=3, ky=3, s=0)
        
        self.J_rotor = self.Irx  # rotor inertia  [Kg.m^2]
        self.R_rotor = 63   # Rotor radius   
        
        #wind speed profile
        height_base=179.9155
        wind_profile_30 = scipy.io.loadmat('wind_profile_30.mat')
        t_interp=wind_profile_30['t_interp'][:,0]
        v_interp=wind_profile_30['v_interp'][:,0]
               
        if wind_type==1:
            v=CubicSpline(t_interp/4,wind_avg*1/14.4607*v_interp*((self.Dr)/height_base)**0.2);
        elif wind_type==2:
            v=lambda t: wind_avg*np.ones((len(t),1))*((self.Dr)/height_base)**0.2
        elif wind_type==3:
            v=lambda t: 5*(1+scipy.signal.square(t*(2*np.pi/30)))+5 #period: 30 , duty: 50  
            
        self.v=v
        self.P_max = 5e6
        
        self.K_feedback_pitch=41.3751
        self.KI=0.008068634
        self.Kp=0.01882681
        self.N_Gear=97
        self.omega_r=1.2671