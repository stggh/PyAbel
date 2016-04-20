# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose
import abel
from abel.benchmark import absolute_ratio_benchmark

DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def test_linbasex_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = abel.linbasex.linbasex_transform(x, direction='inverse')
    assert recon.shape == (n, n) 


def test_linbasex_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    recon = abel.linbasex.linbasex_transform(x, direction="inverse")
    assert_allclose(recon, 0)


def test_linbasex_cyl_gaussian(n=101):
    gauss = lambda r, r0, sigma: np.exp(-(r-r0)**2/sigma**2)

    image_shape=(n, n)
    rows, cols = image_shape
    r2 = rows//2 + rows % 2
    c2 = cols//2 + cols % 2
    sigma = 20*n/100

    x = np.linspace(-c2, c2, cols)
    y = np.linspace(-r2, r2, rows)

    X, Y = np.meshgrid(x, y)

    IM = gauss(X, 0, sigma) # cylindrical Gaussian located at pixel R=0
    Q0, Q1, Q2, Q3 = abel.tools.symmetry.get_image_quadrants(IM)
    ospeed = abel.tools.vmi.angular_integration(Q0, origin=(0, 0))

    # linbasex method inverse Abel transform
    AQ0 = abel.linbasex.linbasex_transform(Q0)
    lspeed = abel.tools.vmi.angular_integration(AQ0, origin=(0, 0))

    ratio_2d = np.sqrt(np.pi)*sigma

    assert_allclose(ospeed[1], lspeed[1]*ratio_2d, rtol=0.0, atol=0.5)

if __name__ == "__main__":
    test_linbasex_shape()
    test_linbasex_zeros()
    test_linbasex_cyl_gaussian()
