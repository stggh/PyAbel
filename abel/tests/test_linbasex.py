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


def test_linbasex_dribinski_image():
    """ Check hansenlaw forward/inverse transform
        using BASEX sample image, comparing speed distributions
    """

    # BASEX sample image
    IM = abel.tools.analytical.sample_image()

    # hansenlaw forward projection
    fIM = abel.Transform(IM, method="hansenlaw", direction="forward").transform

    # inverse Abel transform
    ifIM = abel.Transform(fIM, method="linbasex",
                          transform_options=dict(return_Beta=True))

    # speed distribution
    orig_speed, orig_radial = abel.tools.vmi.angular_integration(IM)
    
    speed = ifIM.linbasex_angular_integration

    orig_speed /= orig_speed[50:125].max()
    speed /= speed[50:125].max()
    import matplotlib.pyplot as plt
    plt.plot(orig_speed, label='orig.')
    plt.plot(speed, label='linbasex')
    plt.legend()
    plt.show()
   

    assert np.allclose(orig_speed[50:125], speed[50:125], rtol=0.5, atol=0)


if __name__ == "__main__":
    test_linbasex_shape()
    test_linbasex_zeros()
    test_linbasex_dribinski_image()
