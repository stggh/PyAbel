# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import abel
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
    r""" Fourier cosine series Abel transform using the algorithm of `G. Pretzler Z. Naturfosch. 46 a, 639-641 (1991) <https://doi.org/10.1515/zna-1991-0715>`_.

    Least-squares fits:

    .. math::

       H(y) = 2\sum_{n=N_l}^{N_u} A_n h_n(y)

    to the image data `IM` to determine the series expansion coeffients :math:`A_n`,
    where:

    .. math::

      h_n(y) = \int_y^R f_n(r) \frac{r}{\sqrt{r^2 - y^2}} dr

    is the standard inverse Abel transform.

    The source distribution is then given by:

    .. math::

      f(r) = \sum_{n=N_l}^{N_u} A_n f_n(r)

    with :math:`f_n(r)` the basis function, a Fourier cosine series:

    .. math ::
    
        f_n(r) = 1 - (-1)^n \cos(n \pi \frac{r}{R})

    and :math:`f_0(r) = 1`.


    Parameters
    ----------
    IM : 1D or 2D numpy array
        Right-side half-image (or quadrant).

    Nl : int
        Lowest coefficient order `n` of the Fourier cosine series:
        :math:`A_n \cos(\pi n r/R)`, typically :math:`N_l = 0`.

    Nu : int
        Uppermost ceofficient of Fourier cosine series.
        A larger value increases the amount of computation, but may improve
        the end transform. Typically, :math:`N_u \sim 4-200` depending on
        the image structure and the number of columns.

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

    # coefficients of cosine series: f(r) = An (1 - (-1)^n cos(n pi r/R))
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
                                                           Nu=Nu,
                                                           direction=direction)

    if transform_IM.shape[0] == 1:
        transform_IM = transform_IM[0]   # flatten to a vector

    return transform_IM


def _fourier_expansion_transform_with_basis(IM, Basis, dr=1, Nu=None,
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

    rows, cols = fbasis.shape

    # Cosine coefficients at unit frequencies
    fftfreq = np.fft.rfftfreq(cols, d=dr/cols)

    unitfreqindx = np.zeros(Nu, dtype=int)
    for i in np.arange(Nu):
        indx =  np.abs(fftfreq-i).argmin()
        unitfreqindx[i] = indx

    # Fourier series coefficients
    An = np.zeros(Nu)

    trans_IM = np.zeros_like(IM)

    for row, imrow in enumerate(IM):
        # duplicate function to make symmetric - coefficients are then real
        fft = np.fft.rfft(np.append(imrow[::-1], imrow)).real/cols

        An[1:] = fft[unitfreqindx[:-1]] 
 
        # Abel transform  Eq. (3) inverse, or Eq. (5) forward
        trans_IM[row] = factor[1]*np.dot(An, fbasis)

    return trans_IM*Jacobian


def f(r, b, n):
    """basis function, a Fourier cosine series Eq(4).

       .. math::

            f_n(r) = 1 - (-1)^n \cos(n \pi \\frac{r}{R})

    Parameters
    ----------
    r : 1D numpy array
        radial grid

    b : float
        maximum radius, usually r[-1]

    n : int
        order of cosine series element

    Returns
    -------
    f : 1D numpy array the same size a `r`
        basis function value(s) for order `n`.
    """

    if n == 0:
        return np.zeros_like(r)

    # return 1 - (-1)**n * np.cos(np.pi*n*r/b)
    return 1 - (1 - 2*(n%2))**n * np.cos(np.pi*n*r/b)


def _fh(r, a, b, n):
    """ Abel transform integrand of f(r), Eq(6).

    """
    return f(r, b, n)*r/np.sqrt(r**2 - a**2)


def _hquad(a, b, n):
    """Abel transform of basis function f(r), h(y) in Eq(6).

    """
    # 1.0e-9 offset to prevent divide by zero
    return quadrature(_fh, a+1.0e-9, b, args=(a, b, n), rtol=1.0e-4,
                      maxiter=500)[0]


def _hgauss(a, b, n, sample_pts, weights):
    """  Abel transform of basis function f(r), h(y) in Eq(6).

    """

    # radii = ((b + a) + (b - a)*sample_pts)/2
    # weights *= np.sqrt(1 - sample_pts**2)
    # return np.dot(weights, _fh(radii, a, b, n))*(b - a)/2

    sumi = 0
    for l, xi in enumerate(sample_pts):
       radius = (b + a)/2 + (b - a)*xi/2
       sumi += weights[l]*_fh(radius, a, b, n)*np.sqrt(1 - xi**2)

    return sumi*(b - a)/2


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

    # Gauss-Chebyshev quadrature
    sample_pts, weights = np.polynomial.chebyshev.chebgauss(Nu)

    r = np.arange(cols)
    R = r[-1]   # maximum radial integration range

    for i, n in enumerate(N):
        fbasis[i] = f(r, R, n)
        for j in r[:-1]:
            hbasis[i, j] = _hquad(j, R, n)
#            hbasis[i, j] = _hgauss(j, R, n, sample_pts, weights)

    return (fbasis, hbasis)
