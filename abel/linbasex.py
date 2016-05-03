# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import scipy as sci
from scipy.special import eval_legendre
from scipy import ndimage

###############################################################################
# linbasex - inversion procedure based on 1-dimensional projections of
#            velocity-map images
#
# As described in:
#   Gerber, Thomas, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi,
#   Peter Radi, and Yaroslav Sych,
#     “Charged Particle Velocity Map Image Reconstruction with One-Dimensional
#      Projections of Spherical Functions.”
#     Review of Scientific Instruments 84, no. 3 (March 1, 2013):
#                                      033101–033101 – 10.
#     doi:10.1063/1.4793404.
#
# 2016-04- Thomas Gerber and Daniel Hickstein - theory and code updates
# 2016-04- Stephen Gibson core code extracted from the supplied jupyter
#          notebook (see #167: https://github.com/PyAbel/PyAbel/issues/167)
#
###############################################################################

_linbasex_parameter_docstring = \
    """Inverse Abel transform using 1d projections of VM-images.

    Gerber, Thomas, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi,
    Peter Radi, and Yaroslav Sych.

    `Charged Particle Velocity Map Image Reconstruction with One-Dimensional
     Projections of Spherical Functions.`
     Review of Scientific Instruments
      84, no. 3 (March 1, 2013): 033101–033101 – 10.
      doi:10.1063/1.4793404.

     linbasex determines the contribution of spherical functions,
     calculating the weight of each Y_i0. The reconstructed 3D object
     is obtained by adding all the Y_i0 contributions, from which
     slices are derived.


    Parameters
    ----------
    Dat: numpy 2D array
        image data must be square shape of odd size
    an: list
        projection angle, in degrees
        e.g. [0, 90] or [0, 54.7356, 90] or [0, 45, 90, 135]
    un: list
        order of Legendre polynomials to be used as the expansion
            even polynomials [0, 2, ...] gerade
            odd polynomials [1, 3, ...] ungerade
            all orders [0, 1, 2, ...].
        In a single photon experiment there are only anisotropies up to
        second order. The interaction of 4 photons (four wave mixing) yields
        anisotropies up to order 8.
    inc: int
        number of pixels per Newton sphere (default 1)
    sig_s: float
        sigma for smoothing (default 0.5)
    rcond: float
        (default 0.0005) scipy.linalg.lstsq fit conditioning value.
        set rcond to zero to switch conditioning off
    threshold: float
        threshold for normalization of higher order Newton spheres
        (default 0.2)
    clip: int
        clip first vectors (smallest Newton spheres) to avoid singularities
    norm_range: tuple 
        (low, high)
        normalization of Newton sphere, maximum in range
        Beta[0, low:high]
    return_Beta: bool
        return the Beta array of Newton spheres
            for the case un=[0, 2]
            Beta0[k] vs k -> speed distribution
            Beta2[k] vs k -> anisotropy of each Newton sphere
    direction: str
        "inverse" - only option for this method.
        Abel transform direction.
    verbose: bool
        print information about processing (normally used for debugging)


    Returns
    -------
    inv_Data: numpy 2D array
       inverse Abel transformed image
    Beta: numpy 2D array  (if return_Beta=True)
       contributions of each spherical harmonic Y_i0 to the 3D distribution.
       contains all information one can get from an experiment.
       for the case un=[0, 2]:
           Beta0[k] vs k -> speed distribution
           Beta2[k] vs k -> anisotropy of each Newton sphere
    projections: numpy 2D array
       Radial projection profiles at angles `an`

    """


def linbasex_transform(Dat, an=[0, 90], un=[0, 2], inc=1, sig_s=0.5,
                       rcond=0.0005, threshold=0.2, basis_dir='.',
                       return_Beta=False, clip=0, norm_range=(0, -1),
                       direction="inverse", verbose=False, **kwargs):
    """wrapper function for linebasex to process supplied quadrant-image
       as a full-image.

    PyAbel transform functions operate on the right side of an image.
    Here we follow the ``basex`` technique of duplicating the right side to
    the left re-forming the whole image.

    """
    Dat = np.atleast_2d(Dat)

    quad_rows, quad_cols = Dat.shape
    full_image = abel.tools.symmetry.put_image_quadrants((Dat, Dat, Dat, Dat),
                           original_image_shape=(quad_rows*2-1, quad_cols*2-1))

    # inverse Abel transform
    recon, Beta, QLz, radial = linbasex_transform_full(full_image, an=an, un=un,
                                inc=inc, sig_s=sig_s, basis_dir=basis_dir,
                                threshold=threshold, clip=clip,
                                norm_range=norm_range,
                                verbose=verbose, **kwargs)

    # unpack right-side
    inv_Dat = abel.tools.symmetry.get_image_quadrants(recon)[0]

    if return_Beta:
        return inv_Dat, Beta, QLz
    else:
        return inv_Dat


def linbasex_transform_full(Dat, an=[0, 90], un=[0, 2], inc=1, sig_s=0.5,
                            rcond=0.0005, threshold=0.2, clip=0, basis_dir='.',
                            return_Beta=False, norm_range=(0, -1),
                            direction="inverse", verbose=False, **kwargs):
    """interface function that fetches/calculates the Basis and
       then evaluates the linbasex inverse Abel transform for the image.

    """

    Dat = np.atleast_2d(Dat)

    rows, cols = Dat.shape

    if rows != cols:
        raise ValueError('image has shape ({}, {}), '.format(rows, cols) +
                         'must be square for a "linbasex" transform')

    # generate basis or read from file if available
    Basis = abel.tools.basis.get_bs_cached("linbasex", cols,
                  basis_dir=basis_dir,
                  basis_options=dict(an=an, un=un, inc=inc, clip=clip,
                                     verbose=verbose))

    return _linbasex_transform_with_basis(Dat, Basis, an=an, un=un, inc=inc,
                                          rcond=rcond, sig_s=sig_s,
                                          threshold=threshold, clip=clip,
                                          norm_range=norm_range)


def _linbasex_transform_with_basis(Dat, Basis, an=[0, 90], un=[0, 2], inc=1,
                                   rcond=0.0005, sig_s=0.5, threshold=0.2,
                                   clip=0, norm_range=(0, -1)):
    """linbasex inverse Abel transform evaluated with supplied basis set Basis.

    """

    Dat = np.atleast_2d(Dat)

    rows, cols = Dat.shape

    # Number of used polynoms
    pol = len(un)

    # How many projections
    proj = len(an)

    QLz = np.zeros((proj, cols))  # array for projections.

    # Rotate and project VMI-image for each angle (as many as projections)
    if an == [0, 90]:
        # If coordinates of the detector coincide with the projection
        # directions unnecessary rotations are avoided, i.e.an=[0, 90] degrees
        QLz[0] = np.sum(Dat, axis=1)
        QLz[1] = np.sum(Dat, axis=0)
    else:
        for i in range(proj):
            Rot_Dat = sci.ndimage.interpolation.rotate(Dat, an[i], axes=(1, 0),
                                                       reshape=False)
            QLz[i, :] = np.sum(Rot_Dat, axis=1)

    # arrange all projections for input into "lstsq"
    bb = np.concatenate(QLz, axis=0)

    Beta = beta_solve(Basis, bb, pol, rcond=rcond)

    inv_Dat, Beta_convol = _Slices(Beta, un, sig_s=sig_s)

    # normalize
    Beta = single_Beta_norm(Beta_convol, threshold=threshold,
                            norm_range=norm_range)

    radial = np.linspace(clip, cols//2, len(Beta[0]))

    return inv_Dat, Beta, QLz, radial

linbasex_transform_full.__doc__ = _linbasex_parameter_docstring


def beta_solve(Basis, bb, pol, rcond=0.0005):
    # set rcond to zero to switch conditioning off

    # array for solutions. len(Basis[0,:])//pol is an integer.
    Beta = np.zeros((pol, len(Basis[0])//pol))

    # solve equation
    Sol = np.linalg.lstsq(Basis, bb, rcond)

    # arrange solutions into subarrays for each β.
    Beta = Sol[0].reshape((pol, len(Sol[0])//pol))

    return Beta


def _SL(i, x, y, Beta_convol, index, un):
    """Calculates interpolated β(r), where r= radius"""
    r = np.sqrt(x**2 + y**2 + 0.1)  # + 0.1 to avoid divison by zero.

    # normalize: divison by circumference.
    BB = np.interp(r, index, Beta_convol[i, :], left=0)/(2*np.pi*r)

    return BB*eval_legendre(un[i], x/r)


def _Slices(Beta, un, sig_s=0.5):
    """Convolve Beta with a Gaussian smoothing function of 1/e width sig_s.

    """

    pol = len(un)
    NP = len(Beta[0])  # number of points in 3_d plot.
    index = range(NP)

    Beta_convol = np.zeros((pol, NP))
    Slice_3D = np.zeros((pol, 2*NP, 2*NP))


    # Convolve Beta's with smoothing function
    if sig_s > 0:
        # smoothing function
        Basis_s = np.fromfunction(lambda i: np.exp(-(i-(NP)/2)**2/(2*sig_s**2))/
                              (sig_s*2.5), (NP,))
        for i in range(pol):
            Beta_convol[i] = np.convolve(Basis_s, Beta[i], mode='same')
    else:
        Beta_convol = Beta

    # Calculate ordered slices:
    for i in range(pol):
        Slice_3D[i] = np.fromfunction(lambda k, l: _SL(i, (k-NP), (l-NP),
                                      Beta_convol, index, un), (2*NP, 2*NP))
    # Sum ordered slices up
    Slice = np.sum(Slice_3D, axis=0)

    return Slice, Beta_convol


def int_beta(Beta, inc=1, threshold=0.1, regions=[(37, 40), (69, 72), (89, 92),
                                                  (133, 136)]):
    """Integrate beta over a range of Newton spheres.

    Parameters
    ----------
    Beta: numpy array
        Newton spheres
    inc: int
        number of pixels per Newton sphere (default 1)
    threshold: float
        threshold for normalisation of higher orders, 0.0 ... 1.0.
    regions: list of tuple radial ranges
        [(min0, max0), (min1, max1), ...]

    Returns
    -------
    Beta_in: numpy array
        integrated normalized Beta array [Newton sphere, region]

    """
    pol = Beta.shape[0]
    # Define new array for normalized beta's, independent of Beat_norm
    Beta_n = np.zeros(Beta.shape)

    # Normalized to Newton sphere with maximal counts.
    max_counts = max(Beta[0, :])

    Beta_n[0] = Beta[0]/max_counts
    for i in range(1, pol):
        Beta_n[i] = np.where(Beta[0]/max_counts > threshold, Beta[i]/Beta[0],
                             0)

    Beta_int = np.zeros((pol, len(regions)))   # arrays for results

    for j, reg in enumerate(regions):
        for i in range(pol):
            Beta_int[i, j] = np.sum(Beta_n[i, range(*reg)])/(reg[1]-reg[0])

    return Beta_int


def single_Beta_norm(Beta, threshold=0.2, norm_range=(0, -1)):
    """Normalize Newton spheres.

    Parameters
    ----------
    Beta: numpy array
        Newton spheres
    threshold: float
        choose only Beta's for which Beta0 is greater than the maximal Beta0
        times threshold in the chosen range
        Set all βi, i>=1 to zero if the associated β0 is smaller than threshold

    norm_range: tuple (int, int)
        (low, high)
        normalize to Newton sphere with maximum counts in chosen range.
        Beta[0, low:high]

    Return
    ------
    Beta: Newton spheres

    """
    pol = Beta.shape[0]

    Beta_norm = np.zeros_like(Beta)
    # Normalized to Newton sphere with maximum counts in chosen range.
    max_counts = Beta[0, norm_range[0]:norm_range[1]].max()

    Beta_norm[0] = Beta[0]/max_counts
    for i in range(1, pol):
        Beta_norm[i] = np.where(Beta[0]/max_counts > threshold,
                                Beta[i]/Beta[0], 0)

    return Beta_norm


def _bas(ord, angle, COS, TRI):
    """Define Basis vectors for a given polynomial order "order" and a
       given projection angle "angle".

    """

    basis_vec = sci.special.eval_legendre(ord, angle) *\
                sci.special.eval_legendre(ord, COS) * TRI
    return basis_vec


def _bs_linbasex(cols, an=[0, 90], un=[0, 2], inc=1, clip=0, **kwargs):

    pol = len(un)
    proj = len(an)

    # Calculation of Base vectors
    # Define triangular matrix containing columns x/y (representing cos(θ)).
    n = cols//2 + cols % 2
    Index = np.indices((n, n))
    Index[:, 0, 0] = 1
    cos = Index[0]*np.tri(n, n, k=0)[::-1, ::-1]/np.diag(Index[0])

    # Concatenate to "bi"-triangular matrix
    COS = np.concatenate((-cos[::-1, :], cos[1:, :]), axis=0)
    TRI = np.concatenate((np.tri(n, n, k=0,)[:-1, ::-1],
                          np.tri(n, n, k=0,)[::-1, ::-1]), axis=0)

    # inc: use only each inc other vector. Keep the base vector with full span
    COS = COS[:, ::-inc]
    TRI = TRI[:, ::-inc]

    COS = COS[:, ::-1]  # rearrange base vectors again in ascending order
    TRI = TRI[:, ::-1]

    if clip > 0:
        # clip first vectors (smallest Newton spheres) to avoid singularities
        COS=COS[:,clip:]
        # It is difficult to trace the effect on the SVD solver used below.
        TRI=TRI[:,clip:]  # usually no clipping works fine.

    # Calculate base vectors for each projection and each order.
    B = np.zeros((pol, proj, len(COS[:, 0]), len(COS[0, :])))

    Norm = np.sum(_bas(0, 1, COS, TRI), axis=0)  # calculate normalization
    cos_an = np.cos(np.radians(an))  # angles in radians

    for p in range(pol):
        for u in range(proj):
            B[p, u] = _bas(un[p], cos_an[u], COS, TRI)/Norm

    # concatenate vectors to one matrix of bases
    Bpol = np.concatenate((B), axis=2)
    Basis = np.concatenate((Bpol), axis=0)

    return Basis

linbasex_transform.__doc__ += _linbasex_parameter_docstring
