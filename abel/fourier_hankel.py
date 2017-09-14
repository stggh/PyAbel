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


def dht(X, d = 1, nu = 0, axis=-1, b = 1):
    N = X.shape[axis]

    m = np.arange(N)
    freq = m/(d*N)

    F = b * jn(nu, np.outer(b*m, np.pi*m/N)) # *m

    return d**2 * np.tensordot(F, X, axes=([1], [axis])) # , freq


def dft(X, axis=-1):
    # discrete Fourier transform

    # Build a slicer to remove last element from flipped array
    slc = [slice(None)] * len(X.shape)
    slc[axis] = slice(None,-1)

    X = np.append(np.flip(X,axis)[slc], X, axis=axis)  # make symmetric
    n = X.shape[axis]
    fftX = np.abs(np.fft.rfft(X, axis=axis))[::2]*2/n

    return fftX


def fourier_hankel_transform(IM, dr=1, direction='inverse', 
                             basis_dir=None, nu=0, axis=-1):
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
    n = IM.shape[axis]

    if direction == 'inverse':
        fftIM = dft(IM, axis=-1)  # Fourier transform
        transform_IM = dht(fftIM, nu=nu, axis=axis)*n/2  # Hankel
    else:
        htIM = dht(IM, nu=nu, axis=axis)
        transform_IM = dft(htIM)

    if transform_IM.shape[0] == 1:
        transform_IM = transform_IM[0]   # flatten to a vector

    return transform_IM
