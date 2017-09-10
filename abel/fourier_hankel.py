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


def transformation_matrix(jz, nu=0):
    # Baddour transformation matrix Eq. (25) JOSA A32, 611-622 (2015)
    b = 2/(jn(nu+1, jz) * jn(nu+1, jz) * jz[-1])

    return b * jn(nu, np.outer(jz, jz / jz[-1]))


def dht(X, nu=0, axis=-1):
    n = X.shape[axis]
    r = np.arange(n)

    # sample space  jz*r[-1]/jz[-1]
    jz = jn_zeros(nu, n+1)
    N = np.abs(jz-r[-1]).argmin()  # set N such that jz[N] ~ R
    jz = jz[:N+1]

    T = transformation_matrix(jz, nu)

    spl = UnivariateSpline(r, X)
    r_sample = jz*r[-1]/jz[-1]
    Xsample = spl(r_sample)

    HXsample = np.tensordot(T, Xsample, axes=([1], [axis]))*r[-1]**2/jz[-1]

    spl = UnivariateSpline(jz, HXsample)

    return spl(r)


def dft(X, axis=-1):
    # discrete Fourier transform
    n = X.shape[axis]

    # Build a slicer to remove last element from flipped array
    slc = [slice(None)] * len(X.shape)
    slc[axis] = slice(None,-1)

    X = np.append(np.flip(X,axis)[slc], X, axis=axis)  # make symmetric

    return np.abs(np.fft.rfft(X, axis=axis)[:n])/n


def hankel_fourier_transform(X, d=1, nu=0, inverse=True, axis=-1):
    n = X.shape[axis]

    if inverse:
        fx = dft(X, axis=axis)  # Fourier transform
        hf = dht(fx, nu=nu, axis=axis)*n  # Hankel
    else:
        hx = dht(X, nu=nu, axis=axis)
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
