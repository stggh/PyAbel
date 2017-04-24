# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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


def fourier_expansion_transform(IM, Nl=0, Nu=None, basis_dir=None,
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

    Returns
    -------
    AIM : 1D or 2D numpy array
        Inverse Abel transform half-image, the same shape as IM

    """

    IM = np.atleast_2d(IM)
    rows, cols = IM.shape   # shape of input quadrant (or half image)
    c2 = cols//2

    # coefficients of cosine series: f(r) = An (1 - (-1)^n cos(n pi r/R))
    # many coefficients An may provide a better fit, but creates more computation
    if Nu is None:
        Nu = cols//4

    N = np.arange(Nl, Nu)
    An = np.ones_like(N)

    # pre-calculate bases
    #fbasis, hbasis = _bs_fourier_series(cols, N)
    (fbasis, hbasis) = abel.tools.basis.get_bs_cached("fourier_expansion", cols,
                                  basis_dir=basis_dir,
                                  basis_options=dict(N=N, rows=rows))

    # array to hold the inverse Abel transform
    AIM = np.zeros_like(IM)

    for rownum, imrow in enumerate(IM):
        # fit basis to an image row
        res = least_squares(residual, An, args=(imrow, rownum, hbasis))

        An = res.x  # store as initial guess for next row fit

        # inverse Abel transform is the source basis function
        # f(r) = \sum_n  An fn(r)
        # evaluated with the row-fitted coefficients An
        AIM[rownum] = np.dot(An, fbasis)

    return AIM


def residual(An, imrow, rownum, Hbasis):
    # least-squares adjust coefficients An
    # difference between image row and the basis function
    return imrow - 2*np.dot(An, Hbasis[:, rownum])


def f(r, R, n):
    """basis function = Fourier cosine series Eq(4).

    """
    return 1 - (1 - 2*(n % 2)) * np.cos(n*np.pi*r/R) if n > 0 else 1


def fh(r, x, R, n):
    """Abel transform integrand of f(r), Eq(6).

    """
    return f(r, R, n)*r/np.sqrt(r**2 - x**2)


def h(x, R, n):
    """Abel transform of basis function f(r), h(y) in Eq(6).

    """
    # Gaussian integration better for 1/sqrt(r^2 - x^2)
    return quadrature(fh, x+1.0e-9, R, args=(x, R, n), rtol=1.0e-4,
                      maxiter=500)[0]


def _bs_fourier_expansion(cols, N, rows=None):
    """Basis calculations.

    f(r) = Fourier cosine series = original distribution
    h(y) = forward Abel transform of f(r)
    """

    fbasis = np.zeros((len(N), cols))
    if rows is None:
        rows = cols
    hbasis = np.zeros((len(N), rows, cols))

    r = np.arange(cols)
    R = r[-1]   # maximum radial integration range

    for i, n in enumerate(N):
        fbasis[i] = f(r, R, n)
        # hbasis[N, row, col]
        for j in r:
            hbasis[i, :, j] = h(j, R, n)

    return (fbasis, hbasis)
