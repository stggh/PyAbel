# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import abel
from scipy.optimize import least_squares
from scipy.integrate import quadrature


#############################################################################
#
# Fourier cosine series method of
#      G. Pretzler "A new method for numerical Abel-inverson"
#                   Z. Naturforsch 46a, 639-641 (1991)
#
# 2017-04 Stephen Gibson - python coded algorithm
#         Dan Hickstein - code improvements
#
#############################################################################


def fourier_expansion_transform(IM, basis_dir='.', Nl=0, Nu=None, dr=1,
                                method='lsq', direction='inverse',
                                return_coefficients=False):
    r""" Fourier cosine series inverse Abel transform using the algorithm of
         `G. Pretzler Z. Naturfosch. 46 a, 639-641 (1991)
         <https://doi.org/10.1515/zna-1991-0715>`_

    Least-squares fits

    .. math::

       H(y) = 2\sum_{n=N_l}^{N_u} A_n h_n(y)

    to the image data 'IM', determing series expansion coeffients :math:`A_n`,
    where

    .. math::

      h_n(y) = \int_y^R f_n(r) \frac{r}{\sqrt{r^2 - y^2} dr

    is the standard inverse Abel transform.

    The source distribution is then given by:
    :math:`f(r) = \sum_{n=N_l}^{N_u} A_n f_n(r)`


    Parameters
    ----------
    IM : 1D or 2D numpy array
        Right-side half-image (or quadrant).

    Nl : int
        Lowest coefficient of Fourier cosine series.

    Nu : int
        Uppermost ceofficient of Fourier cosine series.

    dr : float
        Sampling size (=1 for pixel images), used for Jacobian scaling.

    method : str
        One of 'fft' or 'lsq'.
        Evaluate basis coefficients using fast-Fourier transform 'fft'
        or least-squares fit 'lsq', to the image row.

    direction : str
        Only the `direction="inverse"` transform is currently implemented

    return_coefficients : bool
        Return the coefficients of the basis function.

    Returns
    -------
    inv_IM : 1D or 2D numpy array
        Inverse Abel transform half-image, the same shape as IM.
    
    An : 1D numpy array
        Cosine series coefficients An (n=Nl, ..., Nu) for the last image row.
        If `return_coefficients` is True.

    """

    IM = np.atleast_2d(IM)
    rows, cols = IM.shape

    # coefficients of cosine series: f(r) = An (1 - (-1)^n cos(n pi r/R))
    # A larger number of coefficients, An, may provide a better fit to the
    # row intensity profile, but this creates more computation

    if Nu is None:
        # choose a number that may work and not be too slow!
        if cols > 10:
            Nu = cols//10
        else:
            Nu = cols - 1

    N = np.arange(Nl, Nu)

    # pre-calculate bases
    # basis name fourier_expansion_{cols}_{Nl}_{Nu}.npy
    Basis = abel.tools.basis.get_bs_cached("fourier_expansion",
                             cols, basis_dir=basis_dir,
                             basis_options=dict(Nl=Nl, Nu=Nu))

    inv_IM, An = _fourier_expansion_transform_with_basis(IM, Basis, dr=dr,
                                                     method=method)

    if inv_IM.shape[0] == 1:
        inv_IM = inv_IM[0]   # flatten to a vector

    if return_coefficients:
        return inv_IM, An
    else:
        return inv_IM


def _fourier_expansion_transform_with_basis(IM, Basis, dr=1, method='lsq'):
    fbasis, hbasis = Basis

    n, cols = fbasis.shape
    # Fourier series coefficients
    An = np.ones(n)

    # array to hold the inverse Abel transform
    inv_IM = np.zeros_like(IM)

    for rownum, imrow in enumerate(IM):
        # fit basis to an image row
        if method == 'fft':
            fourier_signal = np.fft.rfft(imrow)*2/cols

            # coefficients thanks to 
            # http://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy
            a0, a = fourier_signal[0].real, fourier_signal[1:-1].real
            a = np.append(a0, a)

            An = a[:An.size]  # coefficients at unit frequencies
            # change odd n cofficients sign to match basis function
            An[2:-1:2] = -An[2:-1:2]

            # fudge factors
            # An[0] = 0 # a0/2
            # An /= np.pi*n
        else:
            res = least_squares(_residual, An, args=(imrow, hbasis))
            An = res.x  # store as initial guess for next row fit

        # inverse Abel transform is the source basis function
        # f(r) = \sum_n  An fn(r)
        # evaluated with the row-fitted coefficients An
        inv_IM[rownum] = np.dot(An, fbasis)

    return inv_IM/dr, An  # dr Jacobian


def _residual(An, imrow, hbasis):
    # least-squares adjust coefficients An
    # difference between image row and the basis function
    return imrow - 2*np.dot(An, hbasis)


def f(r, R, n):
    """basis function = Fourier cosine series Eq(4).

    """
    return 1 - (1 - 2*(n % 2)) * np.cos(2*n*np.pi*r/R) if n > 0 else 1
    # return 1 - (1 - 2*(n % 2)) * np.cos(n*np.pi*r/R) if n > 0 else 1


def fh(r, x, R, n):
    """Abel transform integrand of f(r), Eq(6).

    """
    return f(r, R, n)*r/np.sqrt(r**2 - x**2)


def h(x, R, n):
    """Abel transform of basis function f(r), h(y) in Eq(6).

    """
    # Gaussian integration better for 1/sqrt(r^2 - x^2)
    # 1.0e-9 offset to prevent divide by zero
    return quadrature(fh, x+1.0e-9, R, args=(x, R, n), rtol=1.0e-4,
                      maxiter=500)[0]


def _bs_fourier_expansion(cols, Nl=0, Nu=None):
    """Basis calculations.

    f(r) = Fourier cosine series = original distribution
    h(y) = forward Abel transform of f(r)
    """

    if Nu is None:
        # choose a number that may work and not be too slow!
        if cols > 10:
            Nu = cols//10
        else:
            Nu = cols - 1

    N = np.arange(Nl, Nu)

    fbasis = np.zeros((N.size, cols))
    hbasis = np.zeros((N.size, cols))

    r = np.arange(cols)
    R = r[-1]   # maximum radial integration range

    for i, n in enumerate(N):
        fbasis[i] = f(r, R, n)
        for j in r:
            hbasis[i, j] = h(j, R, n)

    return (fbasis, hbasis)
