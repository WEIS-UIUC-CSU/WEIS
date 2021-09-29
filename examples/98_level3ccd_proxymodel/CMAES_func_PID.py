# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:40:18 2021

@author: bayat2
"""
import numpy as np
import scipy.linalg as la
import concurrent.futures
#from parfor import pmap
from InnerLoop_CMAES_func_PID import Innerloop_CMAES_PID
from WISDEM_func import run_wisdem_tower_design
from os import system

def obj_tst(x,y):
    z=x[0]+x[1]+x[2]+x[3]+y
    return z

def CMAES_Func_PID(obj_func,N,funcval_stop,lambda_pop,x0,OLP_instance):
    # --------------------  Initialization --------------------------------  
    # User defined input parameters (need to be edited)
    #strfitnessfct = obj_func  # name of objective/fitness function
    strfitnessfct = Innerloop_CMAES_PID
    #N = 2;               % number of objective variables/problem dimension
    #xmean = 2*rand(N,1)-1;    % objective variables initial point
    #xmean=zeros(N,1);
    xmean=x0;
    sigma = 0.3;           # coordinate wise standard deviation (step size)
    #stopfitness = 1e-10;  % stop if fitness < stopfitness (minimization)
    #stopeval = 1e3*N^2;   % stop after stopeval number of function evaluations
    stopeval = funcval_stop;
    #stopeval=5;
    
    mu = lambda_pop/2      # number of parents/points for recombination
    weights = np.log(mu+1/2)-np.log(np.arange(mu+1)[1:]) # muXone array for weighted recombination
    mu = np.floor(mu)        
    weights = weights/np.sum(weights)     #normalize recombination weights array
    mueff=np.sum(weights)**2/np.sum(weights**2) # variance-effectiveness of sum w_i x_i
    
    # Strategy parameter setting: Adaptation
    cc = (4+mueff/N) / (N+4 + 2*mueff/N);  # time constant for cumulation for C
    cs = (mueff+2) / (N+mueff+5);  # t-const for cumulation for sigma control
    c1 = 2 / ((N+1.3)**2+mueff);    # learning rate for rank-one update of C
    cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)**2+mueff));  # and for rank-mu update
    damps = 1 + 2*np.max(np.array([0, np.sqrt((mueff-1)/(N+1))-1])) + cs; # damping for sigma 
                                                      # usually close to 1
    # Initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros((N,1)); ps = np.zeros((N,1));   # evolution paths for C and sigma
    B = np.eye(N,N);                       # B defines the coordinate system
    D = np.ones((N,1));                      # diagonal D defines the scaling
    C = B @ np.diag(D.squeeze()**2) @ B.T;            # covariance matrix C
    invsqrtC = B @ np.diag(D.squeeze()**-1) @ B.T;    # C^-1/2 
    eigeneval = 0;                      # track update of B and D
    chiN=N**0.5*(1-1/(4*N)+1/(21*N**2));  # expectation of 
                                        #   ||N(0,I)|| == norm(randn(N,1)) 
                  
    x_1_b=OLP_instance.x_1_b;  x_1_a=OLP_instance.x_1_a;
    x_2_b=OLP_instance.x_2_b;  x_2_a=OLP_instance.x_2_a;
    x_3_b=OLP_instance.x_3_b;  x_3_a=OLP_instance.x_3_a;
    x_4_b=OLP_instance.x_4_b;  x_4_a=OLP_instance.x_4_a;
                
                                        
    # -------------------- Generation Loop --------------------------------
    counteval = 0;  # the next 40 lines contain the 20 lines of interesting code 

    jj=0;
    cond=True;       
    
    x_opt_vector=np.zeros((N,int(stopeval/lambda_pop)))
    f_opt_vector=np.zeros(int(stopeval/lambda_pop))

    while counteval < stopeval and cond:
        #delete(gcp('nocreate'));
        #parpool(lambda*6);
        
        arx=np.zeros((N,lambda_pop))
        for k in np.arange(lambda_pop):
       #   arx[:,k] = xmean.squeeze() + (sigma * B @ (D * np.random.randn(N,1))).squeeze() # m + sig * Normal(0,C) 
          arx[:,k] = xmean.squeeze()
        
          """
        for k in np.arange(lambda_pop):
            if arx[0,k]<(x_1_b+x_1_a)/(x_1_a-x_1_b):
                arx[0,k]=-1
            if arx[1,k]<(x_2_b+x_2_a)/(x_2_a-x_2_b):
                arx[1,k]=-1
            if arx[2,k]<(x_3_b+x_3_a)/(x_3_a-x_3_b):
                arx[2,k]=-1
            if arx[3,k]<(x_4_b+x_4_a)/(x_4_a-x_4_b):
                arx[3,k]=-1"""
        """
        for k in np.arange(lambda_pop):
            if arx[0,k]<-1:
                arx[0,k]=-1
            elif arx[0,k]>1:
                arx[0,k]=1
            if arx[1,k]<-1:
                arx[1,k]=-1
            elif arx[1,k]>1:
                arx[1,k]=1
            if arx[2,k]<-1:
                arx[2,k]=-1
            elif arx[2,k]>1:
                arx[2,k]=1
            if arx[3,k]<-1:
                arx[3,k]=-1  
            elif arx[3,k]>1:
                arx[3,k]=1 """
         
        for i in np.arange(lambda_pop):
            if i==0:
                arx_added=np.c_[arx[:,i],arx[:,i],arx[:,i],arx[:,i],arx[:,i],arx[:,i]]
            else:
                arx_added=np.c_[arx_added,arx[:,i],arx[:,i],arx[:,i],arx[:,i],arx[:,i],arx[:,i]]
                  
            wind_index=np.tile(np.array([[0],[1],[2],[3],[4],[5]]),(lambda_pop,1))      
        
        AEP_added=np.zeros(lambda_pop*6)  
        
        processes=[]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for indx in range(lambda_pop*6):
                p=executor.submit(strfitnessfct,[arx_added[0,indx],arx_added[1,indx],arx_added[2,indx],arx_added[3,indx]],wind_index[indx],OLP_instance)
                #print(p.result())
                processes.append(p)
            #for f in concurrent.futures.as_completed(processes):
                #print(f.result())
        
        for w_index in np.arange(lambda_pop*6):
            AEP_added[w_index] = -processes[w_index].result()-1*(np.any(arx_added[:,w_index]<-1) or np.any(arx_added[:,w_index]>1)); # objective function call
        """
        for w_index in np.arange(lambda_pop*6):
            arfitness_added[w_index] = strfitnessfct(arx_added[:,w_index],wind_index[w_index])+10*(np.any(arx_added[:,w_index]<-1) or np.any(arx_added[:,w_index]>1));""" # objective function call
        
        AEP=np.zeros(lambda_pop)
        for k in np.arange(lambda_pop):
            AEP[k]=AEP_added[(k)*6]+\
                   AEP_added[(k)*6+1]+\
                   AEP_added[(k)*6+2]+\
                   AEP_added[(k)*6+3]+\
                   AEP_added[(k)*6+4]+\
                   AEP_added[(k)*6+5];
                   
            counteval = counteval+1;
            
        for k in np.arange(lambda_pop):
            if AEP[k]<0.1:
                AEP[k]=0.1
        
        arx_unscaled=np.zeros((N,lambda_pop))
        for k in np.arange(lambda_pop):
            arx_unscaled[0,k]=(x_1_b+x_1_a)/2+(x_1_b-x_1_a)/2*(arx[0,k])
            arx_unscaled[1,k]=(x_2_b+x_2_a)/2+(x_2_b-x_2_a)/2*(arx[1,k])
            arx_unscaled[2,k]=(x_3_b+x_3_a)/2+(x_3_b-x_3_a)/2*(arx[2,k])
            arx_unscaled[3,k]=(x_4_b+x_4_a)/2+(x_4_b-x_4_a)/2*(arx[3,k])
        
        Cost=np.zeros(lambda_pop)
        for k in np.arange(lambda_pop):
            Cost[k]=run_wisdem_tower_design(arx_unscaled[3,k], arx_unscaled[0,k], arx_unscaled[1,k], arx_unscaled[2,k])

        system('clear')
        
        arfitness=np.zeros(lambda_pop)
        for k in np.arange(lambda_pop):
            arfitness[k]=Cost[k]/(AEP[k]*1e3) #USD/MWh
            print(arfitness)
            

        arindex=arfitness.argsort() 
        arfitness.sort()
        xold = xmean;
        xmean = arx[:,arindex[np.arange(int(mu))]]@ np.reshape(weights,(-1,1));   # recombination, new mean value
        
        # Cumulation: Update evolution paths
        ps = (1-cs)*ps+np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (xmean-xold) / sigma 
        hsig =np.linalg.norm(ps,2)/np.sqrt(1-(1-cs)**(2*counteval/lambda_pop))/chiN < 1.4 + 2/(N+1);
        pc = (1-cc)*pc+ hsig * np.sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;
 
        # Adapt covariance matrix C
        artmp = (1/sigma) * (arx[:,arindex[np.arange(int(mu))]]-np.tile(xold,(1,int(mu))))
        C = (1-c1-cmu) * C + c1 * (pc*pc.T + (1-hsig) * cc*(2-cc) * C)+ cmu * artmp @ np.diag(weights) @ artmp.T; # plus rank mu update

        # Adapt step size sigma
        sigma = sigma * np.exp((cs/damps)*(np.linalg.norm(ps,2)/chiN - 1)); 
    
        # Decomposition of C into B*diag(D.^2)*B' (diagonalization)
        if counteval - eigeneval > lambda_pop/(c1+cmu)/N/10:  # to achieve O(N^2):
            eigeneval = counteval;
            C = np.triu(C) + np.triu(C,1).T; # enforce symmetry
            D, B = la.eig(C)                # eigen decomposition, B==normalized eigenvectors
            D=D.real
            D=np.reshape(D,(-1,1))
            D = np.sqrt(D);        # D is a vector of standard deviations now
            invsqrtC = B @ np.diag(D.squeeze()**-1) @ B.T;
            
        if jj==0:
            x_total_vec=arx[:,arindex].T
            f_total_vec=    arfitness.T
        else:
            x_total_vec=np.r_[x_total_vec,arx[:,arindex].T]
            f_total_vec=np.r_[f_total_vec,arfitness.T]
      
        x_opt_vector[:,jj]=arx[:, arindex[0]];
        f_opt_vector[jj]=arfitness[0];
        jj=jj+1
        
    xmin = arx[:, arindex[0]]; # Return best point of last iteration.
                               # Notice that xmean is expected to be even
                               # better.
    fmin=arfitness[0]; 
              
    return xmin,fmin,x_opt_vector,f_opt_vector,x_total_vec,f_total_vec