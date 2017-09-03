# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import numpy as np
import abel
from scipy.special import jn, jn_zeros
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


def construct_dht_matrix(N, nu=0, b=1):
    m = np.arange(N)
    k = np.arange(N)

    jz = jn_zeros(nu, N+1)  # spherical Bessel zeros

    """
    # Murray
    b * jn(nu, np.outer(b*m, k/N))

    # Baddour Eq. (19) 
    b = 2/(jz[N] * jn(nu, jz[k])**2)
    Y = jn(nu, np.outer(jz[m], jz[k] / jz[N]))
    Y = b * Y
    """

    # Baddour Eq. (25)
    b = 2/(jn(nu+1, jz[m]) * jn(nu+1, jz[k]) * jz[N])
    T = b * jn(nu, np.outer(jz[m], jz[k] / jz[N]))
    
    return T


def dht(X, d=1, nu=0, axis=-1, b=1):
    N = X.shape[axis]
    F = construct_dht_matrix(N, nu, b)*np.arange(N)

    return d**2 * np.tensordot(F, X, axes=([1], [axis]))


def dft(X, axis=-1):
    # discrete Fourier transform
    n = X.shape[axis]

    # Build a slicer to remove last element from flipped array
    slc = [slice(None)] * len(X.shape)
    slc[axis] = slice(None, -1)

    X = np.append(np.flip(X, axis)[slc], X, axis=axis)  # make symmetric

    return np.abs(np.fft.rfft(X, axis=axis)[:n])/n


def hankel_fourier_transform(X, d=1, nu=0, inverse=True, axis=-1):
    n = X.shape[axis]

    if inverse:
        fx = dft(X, axis=axis)  # Fourier transform
        hf = dht(fx, d=1/(n*d), nu=nu, b=np.pi, axis=axis)*n  # Hankel
    else:
        hx = dht(X, d=1/(n*d), nu=nu, b=np.pi, axis=axis)
        hf = dft(hx)

    return hf


def fourier_hankel_transform(IM, dr=1, inverse=True, basis_dir=None, nu=0):
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
    transform_IM = hankel_fourier_transform(IM, inverse=inverse, nu=nu)

    if transform_IM.shape[0] == 1:
        transform_IM = transform_IM[0]   # flatten to a vector

    return transform_IM
