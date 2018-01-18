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

def test_fourier_hankel_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = abel.fourier_hankel.fourier_hankel_transform(x,
                                direction='inverse')
    assert recon.shape == (n, n) 


def test_fourier_hankel_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    recon = abel.fourier_hankel.fourier_hankel_transform(x, direction="inverse")
    assert_allclose(recon, 0)


def test_fourier_hankel_dft():
    def f(r, nu, a):   # Gaussian transform pair
        return np.exp(-a*r**2)

    n = 501
    rmax = 50
    a = 200/rmax
    nu = 0

    r = np.linspace(0, rmax, n)

    fr = f(r, nu, a)
    fr = np.atleast_2d(fr)

    # double fft
    fftf, freq = abel.fourier_hankel.dft(fr, dr=r[1]-r[0])
    # transform back
    fft2f, freq = abel.fourier_hankel.dft(fftf, dr=freq[1]-freq[0])

    assert_allclose(fr[0], fft2f[0], rtol=0, atol=0.5) 



def test_fourier_hankel_1d_gaussian(n=101):
    gauss = lambda r, r0, sigma: np.exp(-(r-r0)**2/sigma**2)

    rows, cols = n, n
    r2 = rows//2
    c2 = cols//2

    sigma = 20*n/100

    # 1D Gaussian -----------
    r = np.linspace(0, c2-1, c2)

    orig = gauss(r, 0, sigma)
    orig_copy = orig.copy()

    recon = abel.fourier_hankel.fourier_hankel_transform(orig)

    ratio_1d = 1 # np.sqrt(np.pi)*sigma  

    assert_allclose(orig_copy[20:], recon[20:]*ratio_1d, rtol=0.0, atol=0.5)


def test_fourier_hankel_cyl_gaussian(n=101):
    gauss = lambda r, r0, sigma: np.exp(-(r-r0)**2/sigma**2)

    image_shape=(n, n)
    rows, cols = image_shape
    r2 = rows//2
    c2 = cols//2
    sigma = 20*n/100

    x = np.linspace(-c2, c2, cols)
    y = np.linspace(-r2, r2, rows)

    X, Y = np.meshgrid(x, y)

    IM = gauss(X, 0, sigma) # cylindrical Gaussian located at pixel R=0
    Q0 = IM[:r2, c2:] # quadrant, top-right
    Q0_copy = Q0.copy()
    ospeed = abel.tools.vmi.angular_integration(Q0_copy, origin=(0, 0))

    # fourier_hankel method inverse Abel transform
    AQ0 = abel.fourier_hankel.fourier_hankel_transform(Q0)
    ratio_2d = sigma/np.pi/1.1

    assert_allclose(Q0_copy, AQ0*ratio_2d, rtol=0.0, atol=0.3)

if __name__ == "__main__":
    test_fourier_hankel_shape()
    test_fourier_hankel_zeros()
    test_fourier_hankel_1d_gaussian()
    test_fourier_hankel_dft()
    test_fourier_hankel_cyl_gaussian()
