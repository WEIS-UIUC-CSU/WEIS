# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:47:53 2021

@author: bayat2
"""
import numpy as np
from compute_d import compute_d

class params_spar:
    def __init__(self,aa1,aa2,aa3,d1,d2,dtip,l):
        self.aa1=aa1;
        self.aa2=aa2;
        self.aa3=aa3;
        self.d1=d1;
        self.d2=d2;
        self.dtip=dtip;
        self.l=l;        

def Spar_Update_params_func(l_vb):
    
    rho_pb=3.110110612613180e+03       #5000  - 1025
    l_pb=3.967820327063223             #0 - 10 
    rho_vb=1.691729222765398e+03       #1000 - 1025
    rho=7.851578870723508e+03          #paltform steel density  7850 - 8500
    
    nPoints=651                        #inetgration points
    start_points=0                     #platfrom keel
    end_points=130                     #platform freeboard
    x_lin=np.linspace(0,1,nPoints).T
    z=start_points+x_lin*(end_points-start_points) #platform length sweep
    
    aa1=108;        # a faction of Lc defined in Figure 2.8
    aa2=14;         # a faction of Lc defined in Figure 2.8
    aa3=8;          # a faction of Lc defined in Figure 2.8
    d1=9.4;         # Platform base diameter
    d2=6.5;         # tower's base outer diameter == platform's tip outer diameter 

    a3=6.5;
    a4=0.0339;
    l=77.6;

    dtip=a3-a4*l;  #tower tip diamater
    
    params=params_spar(aa1,aa2,aa3,d1,d2,dtip,l)
    d=compute_d(z,params)
    thk=0.05;                          #platform thickness
    Ap_cross_section=np.pi*(d**2-(d-2*thk)**2)/4;  
    Vp_wall=np.trapz(Ap_cross_section,z)       #platform wall volume
    mp_wall_bar=rho*Ap_cross_section;
    mp_wall=np.trapz(mp_wall_bar,z)
    Jp_wall=np.pi/32*((d)**4-(d-2*thk)**4);
    It_wall=np.pi/64*((d)**4-(d-2*thk)**4);
    Ip_wall_z=np.trapz(rho*Jp_wall,z)*1e-6;
    
    zp_wall=1/mp_wall*np.trapz(z*mp_wall_bar,z)
    Ip_wall_x_base=np.trapz(rho*It_wall+mp_wall_bar*z**2,z)
    Ip_wall_y_base=Ip_wall_x_base;
    Ix_wall_center=Ip_wall_x_base-mp_wall*(zp_wall**2);
    Iy_wall_center=Ix_wall_center;
    
    R_pb=d[0]/2-thk;
    V_pb=np.pi*R_pb**2*l_pb;
    m_pb=V_pb*rho_pb;
    Ix_pb_center=1/4*m_pb*R_pb**2+1/12*m_pb*l_pb**2;
    Iy_pb_center=Ix_pb_center;
    Iz_pb_center=1/2*m_pb*R_pb**2*1e-6;
    
    R_vb=R_pb;
    V_vb=np.pi*R_vb**2*l_vb;
    m_vb=V_vb*rho_vb;
    Ix_vb_center=(1/4*m_vb*R_vb**2+1/12*m_vb*l_vb**2)*1e-6;
    Iy_vb_center=Ix_vb_center;
    Iz_vb_center=1/2*m_vb*R_vb**2*1e-6;

    m_totoal=m_vb+m_pb+mp_wall;
    z_bar_total=(zp_wall*mp_wall+l_pb/2*m_pb+(l_pb+l_vb/2)*m_vb)/m_totoal;
    
    Ix_wall_cg=(Ix_wall_center+mp_wall*((zp_wall-z_bar_total)**2))*1e-6;
    Iy_wall_cg=Ix_wall_cg;
    
    Ix_pb_cg=(Ix_pb_center+m_pb*(l_pb/2-z_bar_total)**2)*1e-6;
    Iy_pb_cg=Ix_pb_cg;
    
    Ix_vb_cg=(Ix_vb_center+m_vb*((l_vb/2+l_pb)-z_bar_total)**2)*1e-6;
    Iy_vb_cg=Ix_vb_cg;
    
    Iz_total=Iz_vb_center+Iz_pb_center+Ip_wall_z;
    Ix_total=Ix_vb_cg+Ix_pb_cg+Ix_wall_cg;
    Iy_total=Iy_vb_cg+Iy_pb_cg+Iy_wall_cg;
    
    mp=m_totoal;
    zp=z_bar_total;
    Ipx=Ix_total*1e6;
    Ipy=Iy_total*1e6;
    Ipz=Iz_total*1e6;
    
    return mp,zp,Ipx,Ipy,Ipz