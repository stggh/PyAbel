# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import numpy as np
import abel
from scipy.special import jn
from scipy.ndimage import zoom
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

def construct_dht_matrix(N, nu  = 0, b = 1):
    m = np.arange(0,N).astype('float')
    n = np.arange(0,N).astype('float')
    return b * jn(nu, np.outer(b*m, n/N))


def dht(X, d = 1, nu = 0, axis=-1, b = 1):
    N = X.shape[axis]
    
    prefac = d**2
    m = np.arange(0,N)
    freq = m/(float(d)*N)
    
    F = construct_dht_matrix(N,nu,b)*m
    return prefac * np.tensordot(F, X, axes=([1],[axis])) #, freq


def dft(X):
    # discrete Fourier transform
    n = X.shape[-1]
    X = np.append(X[::-1], X)  # make symmetric

    return np.abs(np.fft.rfft(X)[:n])/n


def hankel_fourier_transform(X, d=1, nu = 0, direction='inverse'):
    n = X.shape[-1]
    if direction == 'inverse':
        fx = dft(X)  # Fourier transform
        hf = dht(fx, d=1/(n*d), nu=nu, b=np.pi) * n / 2  # Hankel transform
    elif direction == 'forward':
        hx = dht(X, d=1/(n*d), nu=nu, b=np.pi) * 2  # Hankel transform
        hf = dft(hx)  # Fourier transform
    else:
        raise ValueError('direction must be either "inverse" or "forward"')

    return hf


def fourier_hankel_transform(IM, dr=1, direction='inverse', basis_dir=None):
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
    rows, cols = IM.shape

    transform_IM = np.zeros_like(IM)

    for i, row in enumerate(IM):
        transform_IM[i] = hankel_fourier_transform(row, direction=direction)

    if transform_IM.shape[0] == 1:
        transform_IM = transform_IM[0]   # flatten to a vector

    return transform_IM
