.. |nbsp| unicode:: 0xA0 
   :trim:

linbasex
========


Introduction
------------

Inversion procedure based on 1-dimensional projections of VM-images as 
described in Gerber et al. [1]. 

How it works
------------

(*from the abstract*)
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

PyAbel python code was extracted from an `ipython3 notebook <https://www.psi.ch/sls/vuv/Station1_IntroEN/Lin_Basex0.7.zip>`


Citation
--------
[1] `Gerber, Thomas, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi, Peter Radi, and Yaroslav Sych, "Charged Particle Velocity Map Image Reconstruction with One-Dimensional Projections of Spherical Functions.” Rev. Sci. Instrum. 84, no. 3, 033101 (2013) <http://dx.doi.org/10.1063/1.4793404>`

