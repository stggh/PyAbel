# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
from scipy.optimize import least_squares
from scipy.integrate import quadrature

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#############################################################################
#
# Fourier cosine series method 
# G. Pretzler "A new method for numerical Abel-inverson"
#  Z. Naturforsch 46a, 639-641 (1991) 
#
# 2017-04 Stephen Gibson - python coded algorithm
#
#############################################################################

def fourier_transform(IM, Nl=0, Nu=21, basis_dir='.', direction='inverse'):
   r""" Fourier cosine series inverse Abel transform using the algorithm of
        `G. Pretzler Z. Naturfosch. 46 a, 639-641 (1991)
        <https://doi.org/10.1515/zna-1991-0715>`_

   Basis function  math:`f(r) = A_n (1-(-1)^n \cos(n \pi r/R)`
   Transform math:`h(y) = \int_R^\infty \frac{f(r) r}{\sqrt{r^2 - y^2)}dr`
   

   Parameters
   ----------
   IM : 1D or 2D numpy array
       right-side half-image (or quadrant)
   
   Nl : int
       lowest coefficient of Fourier cosine series

   Nu : int
       uppermost ceofficient of Fourier cosine series

   Returns
   -------
   AIM : 1D or 2D numpy array
       inverse Abel transform half-image, the same shape as IM

   """
        
   rows, cols = IM.shape

   # coefficients of cosine series: f(r) = An (1 - (-1)^n cos(n pi r/R))
   N = np.arange(Nl, Nu)
   An = np.ones_like(N)

   # precalculate bases
   fbasis, hbasis = _bs_fourier(N, rows, cols)

   # inverse Abel transform 
   AIM = np.zeros_like(IM)

   # fit basis to image, one row at a time
   for rownum, IMrow in enumerate(IM):
        res = least_squares(residual, An, args=(IMrow, rownum, hbasis))
        An = res.x  # use as initial value for next row fit
        AIM[rownum] = np.dot(An, fbasis)

   return AIM
           
def residual(par, IMrow, rownum, Hbasis):
    return IMrow - 2*np.dot(par, Hbasis[:, rownum])

def f(r, R, n):
    return 1 - (1 - 2*(n % 2)) * np.cos(n*np.pi*r/R) if n > 0 else 1

def fh(r, x, R, n):
    return f(r, R, n)*r/np.sqrt(r**2 - x**2)

def h(x, R, n):
    # Gaussian integration better for 1/sqrt(r^2 - x^2)
    return quadrature(fh, x+1.0e-9, R, args=(x, R, n), rtol=1.0e-4,
                      maxiter=500)[0]

def _bs_fourier(N, rows, cols):
    fbasis = np.zeros((len(N), cols))
    hbasis = np.zeros((len(N), rows, cols))

    r = np.arange(cols)
    R = r[-1]   # maximum radial integration range

    for i, n in enumerate(N): 
        fbasis[i] = f(r, R, n)
        # hbasis[N, row, col]
        for j in r:
           hbasis[i, :, j] = h(j, R, n)

    return fbasis, hbasis 


# main -----------------------------------
if __name__ == "__main__":

    # IM = abel.tools.analytical.sample_image(301, name='dribinski')
    m = 301
    m2 = m//2
    x = np.linspace(-m2, m2, m)
    y = np.linspace(-m2, m2, m)
    XX, YY = np.meshgrid(x, y)
    # 2D Gaussian intensity function at image centre
    IM = np.exp(-(XX**2 + YY**2)/m2/10)

    Q = abel.tools.symmetry.get_image_quadrants(IM)
    Q0 = Q[0]  # top right side quadrant
  

    # forward Abel transform
    fQ0 = abel.hansenlaw.hansenlaw_transform(Q0, direction='forward')

    # inverse Abel
    AQ0 = fourier_transform(fQ0, Nl=0, Nu=31)

    radial, speed = abel.tools.vmi.angular_integration(AQ0, origin=(0, 0))
    realradial, realspeed = abel.tools.vmi.angular_integration(Q0,
                                                               origin=(0, 0))

    # graphics ---------------------
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3)
    ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((2, 3), (1, 0))
    ax2 = plt.subplot2grid((2, 3), (1, 1))
    ax3 = plt.subplot2grid((2, 3), (1, 2))
    ax1.imshow(Q0)
    ax1.axis('off')
    ax2.imshow(fQ0)
    ax2.axis('off')
    ax3.imshow(AQ0, vmin=0)
    ax3.axis('off')
    ax0.plot(radial, speed/speed.max(), label='Fourier')
    ax0.plot(realradial, realspeed/realspeed.max(), zorder=0, label='real')
    ax0.legend()
    plt.show()
