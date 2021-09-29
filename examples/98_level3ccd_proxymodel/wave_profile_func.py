# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 11:33:02 2021

@author: bayat2
"""
import numpy as np

def wave_profile_func(t,qq,params):
    Nw=params.Nw
    cc=np.array([params.c])
    ww=np.array([params.ww])
    dw=params.dw
    beta_w=params.beta_w
    Ha_vec=np.array([params.Ha_vec])
    rand_vec=np.array([params.rand_vec])
    
    u_hat=0
    w_hat=0
    u_dothat=0
    w_dothat=0
    Hw=0
    for  i in np.arange(Nw):
        if np.sinh(cc[i]*dw)==-np.inf:
            u_hat=u_hat+ww[i]*Ha_vec[i]*np.cos(ww[i]*t-cc[i]*(qq[4]*np.cos(beta_w)+0*np.sin(beta_w))+2*np.pi*rand_vec[i])\
            *-1
            w_hat=w_hat-ww[i]*Ha_vec[i]*np.sin(ww[i]*t-cc[i]*(qq[4]*np.cos(beta_w)+0*np.sin(beta_w))+2*np.pi*rand_vec[i])\
            *1
            u_dothat=u_dothat-ww[i]*ww[i]*Ha_vec[i]*np.sin(ww[i]*t-cc[i]*(qq[4]*np.cos(beta_w)+0*np.sin(beta_w))+2*np.pi*rand_vec[i])\
            *-1
            w_dothat=w_dothat-ww[i]*ww[i]*Ha_vec[i]*np.cos(ww[i]*t-cc[i]*(qq[4]*np.cos(beta_w)+0*np.sin(beta_w))+2*np.pi*rand_vec[i])\
            *1
        else:
            u_hat=u_hat+ww[i]*Ha_vec[i]*np.cos(ww[i]*t-cc[i]*(qq[4]*np.cos(beta_w)+0*np.sin(beta_w))+2*np.pi*rand_vec[i])\
            *(np.cosh(cc[i]*(qq[5]+dw))/(np.sinh(cc[i]*dw)))
            w_hat=w_hat-ww[i]*Ha_vec[i]*np.sin(ww[i]*t-cc[i]*(qq[4]*np.cos(beta_w)+0*np.sin(beta_w))+2*np.pi*rand_vec[i])\
            *(np.sinh(cc[i]*(qq[5]+dw))/(np.sinh(cc[i]*dw)))
            u_dothat=u_dothat-ww[i]*ww[i]*Ha_vec[i]*np.sin(ww[i]*t-cc[i]*(qq[4]*np.cos(beta_w)+0*np.sin(beta_w))+2*np.pi*rand_vec[i])\
            *(np.cosh(cc[i]*(qq[5]+dw))/(np.sinh(cc[i]*dw)))
            w_dothat=w_dothat-ww[i]*ww[i]*Ha_vec[i]*np.cos(ww[i]*t-cc[i]*(qq[4]*np.cos(beta_w)+0*np.sin(beta_w))+2*np.pi*rand_vec[i])\
            *(np.sinh(cc[i]*(qq[5]+dw))/(np.sinh(cc[i]*dw)))
        Hw=Hw+Ha_vec[i]*np.cos(ww[i]*t-cc[i]*(qq[4]*np.cos(beta_w)+0*np.sin(beta_w))+2*np.pi*rand_vec[i])
  
    vf=np.array([[u_hat*np.cos(beta_w)],[w_hat]])
    vf_dot=np.array([[u_dothat*np.cos(beta_w)],[w_dothat]])
    
    return vf,vf_dot,Hw
    