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
# Fourier method 
# G. Pretzler "A new method for numerical Abel-inverson"
#  Z. Naturforsch 46a, 639-641 (1991) 
#
# 2017-04 Stephen Gibson - python coded algorithm
#
#############################################################################

def fourier_transform(fQ0, basis_dir='.', direction='inverse'):
   rows, cols = fQ0.shape

   N = np.arange(3)
   An = np.ones_like(N)

   R = cols-1

   Fbasis = _bs_fourier(cols, N)

   AfQ0 = np.zeros_like(fQ0)

   for z, fQ0row in enumerate(fQ0[::-1]):
        res = least_squares(residual, An, args=(fQ0row, N, R))
        An = res.x
        AfQ0[z] = np.dot(An, Fbasis)

   return AfQ0[::-1]
           

def f(r, R, n):
    fr = 1 - (-1)**n * np.cos(n*np.pi*r/R)

    # special case n == 0
    if hasattr(n, "__len__"):
        if n[0] == 0:
            fr[0] = 1
    else:
        if n == 0:
            fr = 1
    return fr


def h(par, N, x, R):
    # integration of fn(r) r/sqrt(r^2 - x^2) for  r = x -> R, for each n
    Fint = np.array([fixed_quad(f, x, R, args=(R, n))[0] for n in N])
    return 2*np.dot(par, Fint)


def residual(par, img_row, N, R):
    res = img_row.copy()
    for x in range(img_row.shape[0]-2):
        res[x] -= h(par, N, x, R)
        
    return res

def _bs_fourier(cols, N):

    basis = np.zeros((len(N), cols))
    for n in N: 
        basis[n] = f(np.arange(cols), cols-1, n)

    return basis 


if __name__ == "__main__":

    #IM = abel.tools.analytical.sample_image(301, name='Ominus')
    x = np.linspace(-50, 50, 101)
    y = np.linspace(-50, 50, 101)
    XX, YY = np.meshgrid(x, y)
    IM = np.exp(-(XX**2 + YY**2)/500)

    Q = abel.tools.symmetry.get_image_quadrants(IM)
    Q0 = Q[0]
  

    # forward Abel transform
    fQ0 = abel.hansenlaw.hansenlaw_transform(Q0, direction='forward')

    # inverse Abel
    AQ0 = fourier_transform(fQ0)

    radial, speed = abel.tools.vmi.angular_integration(AQ0, origin=(0, 0))
    realradial, realspeed = abel.tools.vmi.angular_integration(Q0, origin=(0, 0))

    # graphics
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
