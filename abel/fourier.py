# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
from scipy.optimize import least_squares
from scipy.integrate import simps, fixed_quad

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

def fourier_transform(IM, Nl=0, Nu=None, basis_dir='.', direction='inverse'):
   r""" Fourier cosine series inverse Abel transform using the algorithm of
        `G. Pretzler Z. Naturfosch. 46 a, 639-641 (1991)
        <https://doi.org/10.1515/zna-1991-0715>`_

   Paramters
   ---------
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

   # coefficients An (1 - (-1)^n cos(n pi r/R))
   if Nu == None:
       Nu = cols//2
   N = np.arange(Nl, Nu)
   An = np.ones_like(N)

   Fbasis = _bs_fourier(N, cols)

   AIM = np.zeros_like(IM)

   for z, IMrow in enumerate(IM):
        res = least_squares(residual, An, args=(IMrow, N, cols-1))
        An = res.x  # use as initial value for next row fit
        AIM[z] = np.dot(An, Fbasis)

   return AIM
           

def f(r, R, n):
    return 1 - (1 - 2*(n % 2)) * np.cos(n*np.pi*r/R) if n > 0 else 1

def fh(r, R, n, x):
    return f(r, R, n)*r/np.sqrt(r**2 - x**2)

def h(par, R, N, x):
    # integration of fn(r) r/sqrt(r^2 - x^2) for  r = x -> R, for each n
    hint = np.array([fixed_quad(fh, x, R, args=(R, n, x))[0] for n in N])
    return 2*np.dot(par, hint)


def residual(par, img_row, N, R):
    res = img_row.copy()
    for x in range(img_row.shape[0]-2):
        res[x] -= h(par, R, N, x)
        
    return res

def _bs_fourier(N, cols):

    basis = np.zeros((len(N), cols))
    for n in N: 
        basis[n-N[0]] = f(np.arange(cols), cols-1, n)

    return basis 


if __name__ == "__main__":

    IM = abel.tools.analytical.sample_image(301, name='Ominus')
    # x = np.linspace(-50, 50, 101)
    # y = np.linspace(-50, 50, 101)
    # XX, YY = np.meshgrid(x, y)
    # 2D Gaussian intensity function at image centre
    # IM = np.exp(-(XX**2 + YY**2)/50)

    Q = abel.tools.symmetry.get_image_quadrants(IM)
    Q0 = Q[0]  # top right side quadrant
  

    # forward Abel transform
    fQ0 = abel.hansenlaw.hansenlaw_transform(Q0, direction='forward')

    # inverse Abel
    AQ0 = fourier_transform(fQ0, Nl=0, Nu=51)

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
