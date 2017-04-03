# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
from scipy.optimize import least_squares
from scipy.integrate import simps

import matplotlib.pyplot as plt

#############################################################################
#
# Fourier method 
# G. Pretzler "A new method for numerical Abel-inverson"
#  Z. Naturforsch 46a, 639-641 (1991) 
#
# 2017-04-01 Stephen Gibson - python coding of algorithm
#
#############################################################################


def fourier_transform(fQ0, basis_dir='.', direction='inverse'):
   rows, cols = fQ0.shape

   n = np.arange(5)
   An = np.ones_like(n)

   r = np.arange(cols)

   AfQ0 = np.zeros_like(fQ0)
   for y, fQ0row in enumerate(fQ0[1::-1]):
        subr = r > y
        res = least_squares(residual, An, args=(fQ0row[subr], n, r[subr], y))
        AfQ0[y, subr] = np.dot(res.x, f(n, r[subr]))

   return np.array(AfQ0[::-1])
           

def f(n, r):
    # Fourier series
    f = np.zeros((len(n), len(r)))

    for nn in n:
       f[nn] =  1 - (-1)**nn * np.cos(nn*np.pi*r/r[-1])

    # special case n=0
    if n[0] == 0:
        f[0] = 1
 
    return f


def h(n, r, y):
    # integration of fn(r)   r = y -> R
    return simps(f(n, r)*r/np.sqrt(r**2 - y**2), r)


def residual(par, Q0row, n, r, y):
    return Q0row - 2*np.dot(par, h(n, r, y))


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
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
    ax0.imshow(Q0)
    ax0.axis('off')
    ax1.imshow(fQ0)
    ax1.axis('off')
    ax2.imshow(AQ0, vmin=0)
    ax2.axis('off')
    ax3.plot(radial, speed/speed.max())
    ax3.plot(realradial, realspeed/realspeed.max(), zorder=0)
    plt.show()
