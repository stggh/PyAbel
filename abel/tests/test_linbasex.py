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

    recon = abel.linbasex.linbasex_transform_full(x, basis_dir=None)

    assert recon[0].shape == (n+1, n+1)   # NB shape+1


#def test_linbasex_zeros():
#    n = 21
#    x = np.zeros((n, n), dtype='float32')
#
#    recon = abel.linbasex.linbasex_transform(x, basis_dir=None)
#
#    assert_allclose(recon[0], 0)


#def test_linbasex_step_ratio():
#  
#    n = 51
#    r_max = 25
#
#    ref = abel.tools.analytical.GaussianAnalytical(n, r_max, symmetric=True,
#                                                   sigma=10)
#
#    tr = np.tile(ref.abel[None, :], (n, 1)) # make a 2D array from 1D
#
#    recon = abel.Transform(tr, method="linbasex").transform
#
#    recon1d = recon[n//2 + n%2]
#
#    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon1d)
#
#    assert_allclose( ratio , 1.0, rtol=3e-2, atol=0)


if __name__ == "__main__":
    test_linbasex_shape()
#    test_linbasex_zeros()
#    test_linbasex_step_ratio()
