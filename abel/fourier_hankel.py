# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import numpy as np
import abel
from scipy.special import jn, jn_zeros
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt

#############################################################################
#
# Fourier-Hankel method:    FA = H
#
# 2017-08-29
#  see https://github.com/PyAbel/PyAbel/issues/24#issuecomment-325547436
#  Steven Murray - implementation
#  Stephen Gibson, Dan Hickstein - adapted for PyAbel
#
#############################################################################


def Hankel(F, nu=0):
    """ inverse Hankel transform basic \sum r_i F_i J_nu(2pi j i/2n)

    Based on Whitaker C-code in "Image reconstruction: The Abel transform" Ch 5
        
    """
    n = F.shape[-1]

    Nyquist = 1/(2*n)

    f = np.zeros_like(F)
    i = np.arange(n)

    for j in i:
       q = Nyquist*j
       f[:] += q*F[j]*jn(nu, 2*np.pi*q*i[:])

    return f


def dht(X, nu=0, axis=-1):

    HX = np.zeros_like(X)

    for i, row in enumerate(X):
       HX[i] = Hankel(row, nu=nu)
    
    return HX
        

def dft(X, axis=-1):
    # discrete Fourier transform
    n = X.shape[axis]

    # Build a slicer to remove last element from flipped array
    slc = [slice(None)] * len(X.shape)
    slc[axis] = slice(None,-1)

    X = np.append(np.flip(X,axis)[slc], X, axis=axis)  # make symmetric

    return np.abs(np.fft.rfft(X, axis=axis)[:n])/n


def hankel_fourier_transform(X, d=1, nu=0, direction='inverse', axis=-1):
    n = X.shape[axis]

    if direction == 'inverse':
        import matplotlib.pyplot as plt
        fx = dft(X, axis=axis)  # Fourier transform
        hf = dht(fx, nu=nu, axis=axis)*n  # Hankel
    else:
        hx = dht(X, nu=nu, axis=axis)
        hf = dft(hx)

    return hf


def fourier_hankel_transform(IM, dr=1, direction='inverse', 
                             basis_dir=None, nu=0):
    """
    Parameters
    ----------
    IM : 1D or 2D numpy array
        Right-side half-image (or quadrant).
    dr : float
        Sampling size (=1 for pixel images), used for Jacobian scaling.
    direction : str
        'inverse' or 'forward' Abel transform
    Returns
    -------
    trans_IM : 1D or 2D numpy array
        Inverse or forward Abel transform half-image, the same shape as IM.
    """
    IM = np.atleast_2d(IM)
    transform_IM = hankel_fourier_transform(IM, direction=direction, nu=nu)

    if transform_IM.shape[0] == 1:
        transform_IM = transform_IM[0]   # flatten to a vector

    return transform_IM
