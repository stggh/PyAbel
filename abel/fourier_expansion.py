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
                                direction='inverse'):
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
        A largere value increases the amount of computation, but may improve
        the end transform.

    dr : float
        Sampling size (=1 for pixel images), used for Jacobian scaling.

    direction : str
        'inverse' or 'forward' Abel transform

    Returns
    -------
    trans_IM : 1D or 2D numpy array
        Inverse or forward Abel transform half-image, the same shape as IM.
    
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

    transform_IM = _fourier_expansion_transform_with_basis(IM, Basis, dr=dr,
                                                           direction=direction)

    if transform_IM.shape[0] == 1:
        transform_IM = transform_IM[0]   # flatten to a vector

    return transform_IM


def _fourier_expansion_transform_with_basis(IM, Basis, dr=1,
                                            direction='inverse'):
    if direction == 'forward':
        # swap bases 
        hbasis, fbasis = Basis
        Jacobian = dr
        factor = (1, 2)
    else:
        fbasis, hbasis = Basis
        Jacobian = 1/dr
        factor = (2, 1)

    n, cols = fbasis.shape
    c2 = cols//2

    # Fourier series coefficients - starting values = 1
    An = np.ones(n)

    # image np.array to hold the inverse Abel transform
    trans_IM = np.zeros_like(IM)

    # least-squares fit basis function directly to row intensity profile
    for rownum, imrow in enumerate(IM):
        res = least_squares(_residual, An, args=(imrow, hbasis, factor[0]))
        An = res.x  # keep as the initial guess for next row fit

        # Abel transform  Eq. (3) inverse, or Eq. (5) forward
        trans_IM[rownum] = factor[1]*np.dot(An, fbasis)

    return trans_IM*Jacobian


def _residual(An, imrow, basis, factor=2):
    # least-squares adjust coefficients An
    # difference between image row and the basis function
    return imrow - factor*np.dot(An, basis)


def f(r, R, n):
    """basis function = Fourier cosine series Eq(4).

    """
    if n == 0:
        return np.zeros_like(r)

    return 1 - ((-1)**n) * np.cos(np.pi*n*r/R)


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
