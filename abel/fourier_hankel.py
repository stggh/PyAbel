# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import numpy as np
import abel
from scipy.special import jn

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

def dht(X, dr=1, nu=0, axis=-1, b=1):
    # discrete Hankel transform
    N = X.shape[axis]

    n = np.arange(N)

    freq = n/dr/N

    F = b * jn(nu, np.outer(b*n, n/N)) * n

    return dr**2 * np.tensordot(X, F, axes=([1], [axis])), freq


def dft(X, dr=1, axis=-1):
    # discrete Fourier transform

    # Build a slicer to remove last element from flipped array
    slc = [slice(None)] * len(X.shape)
    slc[axis] = slice(None, -1)

    X = np.append(np.flip(X, axis)[slc], X, axis=axis)  # make symmetric

    fftX = np.abs(np.fft.rfft(X, axis=axis)) * dr
    freq = np.fft.rfftfreq(X.shape[axis], d=dr)

    return fftX, freq


def fourier_hankel_transform(IM, dr=1, direction='inverse', 
                             basis_dir=None, nu=0, axis=-1):
    """
    Parameters
    ----------
    IM : 1D or 2D numpy array
        Right-side half-image (or quadrant)

    dr : float
        Sampling size (=1 for pixel images), used for Jacobian scaling

    direction : str
        'inverse' or 'forward' Abel transform

    Returns
    -------
    trans_IM : 1D or 2D numpy array
        Inverse or forward Abel transform half-image, the same shape as IM.
    """

    IM = np.atleast_2d(IM)

    if direction == 'inverse':
        fftIM, freq = dft(IM, dr=dr, axis=axis)  # Fourier transform
        # Hankel
        transform_IM, freq = dht(fftIM, dr=freq[1]-freq[0], nu=nu, axis=axis)
    else:
        htIM, freq = dht(IM, dr=dr, nu=nu, axis=axis)  # Hankel
        transform_IM, freq = dft(htIM, dr=freq[1]-freq[0])  # fft
        freq *= 2*np.pi

    if transform_IM.shape[0] == 1:
        transform_IM = transform_IM[0]   # flatten to a vector

    return transform_IM, freq
