import numpy as np
from numpy.matlib import repmat
from copy import deepcopy
from smt.sampling_methods import LHS
from scipy.special import comb
from scipy.optimize import minimize

class MaxProESE(LHS):
    
    def __init__(self, **kwargs):
        kwargs['criterion'] = 'ese'
        super(MaxProESE, self).__init__(**kwargs)

    def _ese(self, dim, nt):
        P = super()._ese(dim, nt)
        return self._optimize_MaxPro(P)

    def _PhiP_obj(self, x, n, dim):
        X = x.reshape((n, dim))
        return self._PhiP(X)

    def _optimize_MaxPro(self, P):
        n, dim = P.shape
        xlimits = self.options['xlimits']
        x0 = P.reshape((n*dim))
        bounds = [tuple(it) for it in repmat(xlimits, n, 1).tolist()]

        res = minimize(self._PhiP_obj, x0, args=(n, dim), method='trust-constr', bounds=bounds, options={'verbose': 3, 'disp': True, 'maxiter': 150})
        return res.x.reshape((n, dim))

    def pdiff(self, X):
        X = np.array(X)
        
        n, m = X.shape
        out = np.zeros(shape=((n*(n-1))//2, m), dtype=float)
        
        kdx = 0
        for idx in range(0, n-1):
            for jdx in range(idx+1, n):
                out[kdx, :] = X[idx, :] - X[jdx, :]
                kdx += 1
                
        return out

    def cdiff(self, XA, XB):
        XA = np.array(XA)
        XB = np.array(XB)
        
        nA, mA = XA.shape
        nB, mB = XB.shape
        out = np.zeros(shape=(nA, nB, mA), dtype=float)
        
        for idx in range(0, nA):
            for jdx in range(0, nB):
                out[idx, jdx, :] = XB[jdx, :] - XA[idx, :]
                
        return out
    
    def _PhiP(self, X, p=10):
        return (1.0/np.prod(self.pdiff(X)**2.0, axis=1)).sum()/comb(X.shape[0],2)
    
    def _PhiP_exchange(self, X, k, PhiP_, p, fixed_index):
        # Choose two (different) random rows to perform the exchange
        i1 = self.random_state.randint(X.shape[0])
        while i1 in fixed_index:
            i1 = self.random_state.randint(X.shape[0])

        i2 = self.random_state.randint(X.shape[0])
        while i2 == i1 or i2 in fixed_index:
            i2 = self.random_state.randint(X.shape[0])

        X_ = np.delete(X, [i1, i2], axis=0)

        diff1 = np.squeeze(self.cdiff([X[i1, :]], X_))
        diff2 = np.squeeze(self.cdiff([X[i2, :]], X_))
        
        Xi1 = deepcopy(X[i1, :])
        Xi2 = deepcopy(X[i2, :])
        Xi1[k] = X[i2, k]
        Xi2[k] = X[i1, k]
        
        diffR1 = np.squeeze(self.cdiff([Xi1], X_))
        diffR2 = np.squeeze(self.cdiff([Xi2], X_))
        
        res = (
            PhiP_*comb(X.shape[0],2) \
                - (1.0/np.prod(diff1**2, axis=1)).sum() \
                - (1.0/np.prod(diff2**2, axis=1)).sum() \
                + (1.0/np.prod(diffR1**2, axis=1)).sum() \
                + (1.0/np.prod(diffR2**2, axis=1)).sum()
        )/comb(X.shape[0],2)
        
        X[i1, :] = Xi1
        X[i2, :] = Xi2

        return res
