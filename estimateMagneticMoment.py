# -*- coding: utf-8 -*-
"""
Estimate the magnetic moment of a current configuration

Created on Sun Apr  4 10:59:11 2021

@author: Derick Canceran
@email: dbcanceran@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sop

# Constants
mu_0 = 4*np.pi*10**(-7)

#%% MagDipField
def magDipField(r, m):
    '''
    Calculate the magnetic field of a magnetic dipole.

    Parameters
    ----------
    r : Position vector, numpy array
        Must be a length 3 numpy array in rectangular coordinates.
        Must be a row vector. If multiple sample points, vector must
        be stacked vertically, i.e. shape must be 3 by N
    m : Magnetic dipole moment vector, numpy array
        Must be a length 3 numpy array in rectangular coordinates.

    Returns
    -------
    Magnetic field vector, numpy array

    '''
    
    return 1e9*1e-7*(3*r*np.dot(r,m.T)/(r**2).sum(axis=-1, keepdims=1) - m)/ \
        np.sqrt((r**2).sum(axis=-1, keepdims=1))**3
#%%
m = np.array([-1.5,10,1.0]).reshape([-1,3])
r = np.array([[0,0,1], [0,1,0], [np.sqrt(2),0,0], [1,1,0.0]]).reshape([-1,3])
f = magDipField(r, m)
magn = lambda vec: np.sqrt((vec**2).sum(axis=-1,keepdims=1))
normf = magn(f)
witherrf = f*(1+np.random.normal(size=f.shape)*0.05) #5% deviation normaldist

#%% Residuals

def residuals(m, xdata, ydata):
    '''
    Calculate the residuals of a magnetic moment estimate.

    Parameters
    ----------
    m : array_like with shape (n,)
        magnetic moment estimate
    xdata : position matrix with rows as samples
        DESCRIPTION.
    ydata : measurment data
        DESCRIPTION.

    Returns
    -------
    residuals with shape (m,)

    '''
    
    mest = np.array(m).reshape([-1,3])
    yest = magDipField(xdata, mest)
    return (ydata-yest).ravel()

#%%
result = sop.least_squares(residuals,[1,1,1],args=(r,witherrf),verbose=1,method='lm')
print(np.sqrt((((result.x-m)/m)**2).sum()))