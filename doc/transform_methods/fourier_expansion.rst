Fourier series
==============


Introduction
------------
Fourier cosine series inverse Abel transform using the algorithm of
G. Pretzler [1]. See discussion issue #198 [2].

From the abstract:

*The unknown radial distribution is expanded in a series of cosine-functions. the amplitudes of which are calculated by least-squares-fitting of the Abel-transformed series to the measured data.*

Note this is one of the slowest inverse Abel transform methods, but it has the advantage of tailoring a basis function to the particular image particle distribution.

How it works
------------

Fits each image row to

    .. math::

      H(y) = 2 \sum_{n=N_l}^{N_u} A_n \int_y^R f_n(r) \frac{r}{\sqrt{r^2 - y^2} dr

    to determine coefficients :math:`A_n`.

    The inverse Abel transform image is given by:

    .. math::

      f(r) = \sum_{n=N_l}^{N_u} A_n f_n(r)

    where the basis function  :math:`f(r) = A_n (1-(-1)^n \cos(n \pi r/R)`


When to use it
--------------

Ref. [2] claims: 

*Accuracies of the discrete Fourier expansion (FE) and Fourier Hankel (FH) methods have been compared and analyzed. The FE method is very accurate even for a small set of data, while the FH method appears to have systematic negative errors.
:
Though the FE method has a high accuracy, it is very sensitive to noise and the inversion matrix is
difficult to calculate, which restricts its application; a smoothing procedure must be implemented when it has been used. The FH method can only be considered when applying it to data with a large number of points ...*


Ref. [1] Conclusions:

*The new method is non-iterative, derivative-free, and adaptable to any special problem of this kind.*

This method is works better with low noise data.

How to use it
-------------

To complete the inverse transform of a full image with the ``fourier_expansions``, simply use the :class:`abel.Transform` class: ::

    abel.Transform(myImage, method='fourier_expansion', direction='inverse').transform

Note that the forward `fourier_expansion` Abel transform is not yet implemented in PyAbel.

If you would like to access the `fourier_expansion` algorithm directly (to transform a right-side half-image), you can use :func:`abel.fourier_expansion.fourier_expansion_transform()`.


Example
-------

.. plot:: ../examples/example_fourier_expansion.py
    :include-source:



Notes
-----

The basis coefficients :math:`A_n` are determined by a least-squares fit to each row. Reducing the number of coefficients, smaller `Nl`, will improve execution speed. 




Citation
--------
[1] `G. Pretzler Z. Naturfosch. 46 a, 639-641 (1991) <https://doi.org/10.1515/zna-1991-0715>`_

[2] Discussion `issue #198 <https://github.com/PyAbel/PyAbel/issues/198>`_

[3] Comparison of Fourier expansion vs Fourier Hankel `<https://doi.org/10.1364/AO.47.001350>`_
