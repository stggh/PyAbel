.. |nbsp| unicode:: 0xA0 
   :trim:

Lin basex
=========


Introduction
------------

Inversion procedure based on 1-dimensional projections of VM-images as 
described in Gerber et al. [1]. 

[*from the abstract*]

VM-images are composed of projected Newton spheres with a common centre. 
The 2D images are usually evaluated by a decomposition into base vectors each
representing the 2D projec- tion of a set of particles starting from a centre 
with a specific velocity distribution. We propose to evaluate 1D projections of
VM-images in terms of 1D projections of spherical functions, instead. 
The proposed evaluation algorithm shows that all distribution information can 
be retrieved from an adequately chosen set of 1D projections, alleviating the 
numerical effort for the interpretation of VM-images considerably. The obtained
results produce directly the coefficients of the involved spherical functions, 
making the reconstruction of sliced Newton spheres obsolete.

How it works
------------

[*extract of a comment made by Thomas Gerber (method author)*]

Imaging an PES experiment which produces electrons that are distributed on the 
surface of a sphere. This sphere can be described by spherical functions. If 
all electrons have the same energy we expect them on a (Newton) sphere with 
radius :math:`i`. This radius is projected to the CCD. The distribution on 
the CCD has (if optics are approriate) the same radius :math:`i`. 
Now let us assume that the distribution on the Newton sphere has some 
anisotropy. We can describe the 
distribution on this sphere by spherical functions :math:`Y_{nm}`. 
Lets say :math:`xY_{00} + yY_{20}`. 
The 1D projection of those spheres produces just :math:`xP_{i0}(k) +yP_{i2}(k)`
where :math:`P_{i}` denotes Legendre Polynomials scaled to the interval 
:math:`i` and :math:`k` is the argument (pixel).

For one projection Lin Basex now solves for the parameters :math:`x` and 
:math:`y`.  If we look at another projection turned by an angle, the Basis 
:math:`P_{i0}` and :math:`P_{i2}` 
has to be modified because the projection of e.g., :math:`Y_{20}` turned 
by an angle 
yields another function. It was shown that this function for e.g., 
:math:`P_{2}` is just 
:math:`P_{2}(a)P_{i2}(k)` where :math:`a` is the turning angle. Solving 
the equations for the 1D 
projection at angle (:math:`a`) with this modified basis yields the same 
:math:`x` and :math:`y` 
parameters as before.


When to use it
--------------

tbd

How to use it
-------------

To complete the inverse Abel transform of a full image with the 
``linbasex method``, simply use the :class:`abel.Transform`: class ::

    abel.Transform(myImage, method='linbasex').transform


If you would like to access the linbasex algorithm directly (to transform a 
right-side half-image), you can use :func:`abel.linbasex.linbasex_transform`.


Example
-------

.. plot:: ../examples/example_linbasex.py

Historical
----------

PyAbel python code was extracted from the following `ipython3 notebook <https://www.psi.ch/sls/vuv/Station1_IntroEN/Lin_Basex0.7.zip>`_ supplied by Thomas Gerber.


Citation
--------
[1] `Gerber, Thomas, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi, Peter Radi, and Yaroslav Sych, "Charged Particle Velocity Map Image Reconstruction with One-Dimensional Projections of Spherical Functions.” Rev. Sci. Instrum. 84, no. 3, 033101 (2013) <http://dx.doi.org/10.1063/1.4793404>`_

