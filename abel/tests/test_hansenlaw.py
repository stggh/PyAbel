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

def test_hansenlaw_basis_sets_cache():
    n = 121

    file_name = os.path.join(DATA_DIR, "{}_basis_{}_{}.npz".
                             format("hansenlaw", n, n))

    if os.path.exists(file_name):
        os.remove(file_name)
    # 1st call generate and save
    abel.tools.basis.get_bs_cached(method="hansenlaw", cols=n, basis_dir=DATA_DIR, 
                             verbose=False)
    # 2nd call load from file
    abel.tools.basis.get_bs_cached(method="hansenlaw", cols=n, basis_dir=DATA_DIR, 
                             verbose=False)

    # clean-up
    if os.path.exists(file_name):
        os.remove(file_name)

def test_hansenlaw_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = abel.hansenlaw.hansenlaw_transform(x, direction='inverse')
    assert recon.shape == (n, n) 


def test_hansenlaw_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    recon = abel.hansenlaw.hansenlaw_transform(x, direction="inverse")
    assert_allclose(recon, 0)

def test_hansenlaw_1d_gaussian(n=101):
    gauss = lambda r, r0, sigma: np.exp(-(r-r0)**2/sigma**2)

    rows, cols = n, n
    r2 = rows//2 + rows % 2
    c2 = cols//2 + cols % 2

    sigma = 20*n/100

    # 1D Gaussian -----------
    r = np.linspace(0, c2-1, c2)

    orig = gauss(r, 0, sigma)

    recon = abel.hansenlaw.hansenlaw_transform(orig)

    ratio_1d = np.sqrt(np.pi)*sigma

    assert_allclose(orig[20:], recon[20:]*ratio_1d, rtol=0.0, atol=0.5)

def test_hansenlaw_cyl_gaussian(n=101):
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
    Q0 = IM[:r2, c2:] # quadrant, top-right
    Q0_copy = Q0.copy()
    ospeed = abel.tools.vmi.angular_integration(Q0_copy, origin=(0, 0))

    # hansenlaw method inverse Abel transform
    AQ0 = abel.hansenlaw.hansenlaw_transform(Q0)
    ratio_2d = np.sqrt(np.pi)*sigma

    assert_allclose(Q0_copy, AQ0*ratio_2d, rtol=0.0, atol=0.3)

if __name__ == "__main__":
    test_hansenlaw_shape()
    test_hansenlaw_zeros()
    test_hansenlaw_1d_gaussian()
    test_hansenlaw_cyl_gaussian()
