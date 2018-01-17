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


def Baddour(jz, nu=0):
    # Baddour transformation matrix Eq. (25) JOSA A32, 611-622 (2015)
    b = 2/(jn(nu+1, jz) * jn(nu+1, jz) * jz[-1])

    return b * jn(nu, np.outer(jz, jz / jz[-1]))


def sample_space(r, rho, nu):
    # sample size, choose N so that r x rho = jz[-1]
    jz = jn_zeros(nu, r.size*rho.size)
    N = np.abs(jz-r[-1]*rho[-1]).argmin()  # set N such that jz[N] ~ R
    jz = jz[:N+1]
    return jz


def dhtB(r, func, a, nu=0, axis=-1):

    # sample space  r x rho = jz[-1]

    jz = sample_space(r, r, nu=nu)

    r_sample = jz*r[-1]/jz[-1]

    X = func(r_sample, a)

    T = Baddour(jz, nu)

    return jz, np.tensordot(T, X, axes=([1], [axis]))*r[-1]**2/jz[-1]


def Whitaker(F, nu=0):
    """ inverse Hankel transform basic \sum r_i F_i J_nu(2pi r_i i/2n)

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


def dhtW(X, nu=0, axis=-1):
    HX = np.zeros_like(X)

    for i, row in enumerate(X):
        HX[i] = Whitaker(row, nu=nu)

    return HX


def dht(X, dr=1, nu=0, axis=-1, b=1):
    N = X.shape[axis]

    m = np.arange(N)
    n = np.arange(N)

    freq = m/(dr*N)

    F = b * jn(nu, np.outer(b*m, n/N))*m

    return dr**2 * np.tensordot(X, F, axes=([1], [axis])), freq


def dft(X, dr=1, axis=-1):
    # discrete Fourier transform

    # Build a slicer to remove last element from flipped array
    slc = [slice(None)] * len(X.shape)
    slc[axis] = slice(None,-1)

    X = np.append(np.flip(X,axis)[slc], X, axis=axis)  # make symmetric
    n = X.shape[axis]
    fftX = np.abs(np.fft.rfft(X, axis=axis))*dr
    freq = np.fft.rfftfreq(n, d=dr)

    return fftX, freq


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

    import matplotlib.pyplot as plt

    if direction == 'inverse':
        fftIM, freq = dft(IM, dr=dr, axis=-1)  # Fourier transform
        dr = freq[1] - freq[0]
        transform_IM, freq = dht(fftIM, dr=dr, nu=nu, axis=axis) # Hankel
        transform_IM *= n/2
    else:
        htIM, freq = dht(IM, nu=nu, axis=axis)
        transform_IM, freq = dft(htIM, dr=freq[1]-freq[0])
        transform_IM /= n

    if transform_IM.shape[0] == 1:
        transform_IM = transform_IM[0]   # flatten to a vector

    return transform_IM #, freq
