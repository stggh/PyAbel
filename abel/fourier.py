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


def fourier_transform(Q0, basis_dir='.', direction='inverse'):
   rows, cols = Q0.shape

   n = np.arange(11)
   An = np.ones_like(n)

   r = np.arange(cols)

   AQ0 = np.zeros_like(Q0)
   for y, Q0row in enumerate(Q0[::-1]):
        res = least_squares(residual, An, args=(Q0row, n, r, y))
        AQ0[y] = np.dot(res.x, f(n, r))

   return np.array(AQ0[::-1])
           

def f(n, r):
    f = np.zeros((len(n), len(r)))

    for nn in n:
       if nn == 0:
           f[0] = 1
       else:
           f[nn] =  1 - (1-2*(nn % 2)) * np.cos(nn*np.pi*r/r[-1])
 
    return f

def h(n, r, y):
    return simps(f(n, r)*r/np.sqrt(r**2 - y**2), r)

def residual(par, Q0row, n, r, y):
    res = np.zeros_like(Q0row)
    subr = r > y
    xx = 2*np.dot(par, h(n, r[subr], y))
    print("---", xx.shape)
    return Q0row - (res[subr] + xx)


# -------
if __name__ == "__main__":

    IM = abel.tools.analytical.sample_image(501, name='Ominus')
    Q = abel.tools.symmetry.get_image_quadrants(IM)
    Q0 = Q[0]
    fQ0 = abel.hansenlaw.hansenlaw_transform(Q0, direction='forward')

    AQ0 = fourier_transform(fQ0)
    fig, ax0, ax1, ax2 = plt.subplots(1, 3)
    ax0.imshow(Q0)
    ax1.imshow(fQ0)
    ax2.imshow(AQ0)
    plt.show()
