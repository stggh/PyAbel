# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import abel
import hankel
from scipy.interpolate import RectBivariateSpline

#############################################################################
#
# Fourier-Hankel method:    FA = H
#
# NB uses hankel  https://github.com/steven-murray/hankel
#
# 2017-07 Stephen Gibson - python coded algorithm
#         Dan Hickstein - code improvements
#
#############################################################################


def fourier_hankel_transform(IM, dr=1, direction='inverse', basis_dir=None):
    """
    Parameters
    ----------
    IM : 1D or 2D numpy array
        Right-side half-image (or quadrant).

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

    y = np.arange(rows)
    x = np.arange(cols)
    sp = RectBivariateSpline(y, x, IM)

    grid = np.indices((rows, cols))

    ht = hankel.HankelTransform()
    ft = hankel.SymmetricFourierTransform()
    
    iht = ht.transform(sp.eval, grid, inverse=True) 
    fft = ft.transform(iht, grid)

    transform_IM = iht*fft
  

    if transform_IM.shape[0] == 1:
        transform_IM = transform_IM[0]   # flatten to a vector

    return transform_IM

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    IM = np.loadtxt("O2-ANU1024.txt")
    rows, cols = IM.shape

    IMc = abel.tools.symmetry.get_image_quadrants(IM)
    
    AIM = fourier_hankel_transform(IM[1]) 

    plt.imshow(AIM)
    plt.show()
