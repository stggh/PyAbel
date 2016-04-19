# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy as sci
from scipy.special import eval_legendre
from scipy import ndimage

################################################################################
# linbasex - inversion procedure bases on 1-dimensional projections of 
#            VM-images 
# 
# as described in:
# Gerber, Thomas, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi, 
# Peter Radi, and Yaroslav Sych.
# 
# “Charged Particle Velocity Map Image Reconstruction with One-Dimensional
#  Projections of Spherical Functions.” 
# Review of Scientific Instruments 
#     84, no. 3 (March 1, 2013): 033101–033101 – 10. 
#     doi:10.1063/1.4793404.
# 
# 2016-04-20 Stephen Gibson extract core code from supplied ipython notebook
#
################################################################################

def linbasex_transform(Dat, n=4, an=[0, 90], un=[0, 2], inc=1,
                       verbose=False): 
    """Inverse Abel transform using 1d projections of VM-images.

    Gerber, Thomas, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi, 
    Peter Radi, and Yaroslav Sych.
  
    “Charged Particle Velocity Map Image Reconstruction with One-Dimensional
     Projections of Spherical Functions.” 
     Review of Scientific Instruments 
      84, no. 3 (March 1, 2013): 033101–033101 – 10. 
      doi:10.1063/1.4793404.

    Parameters
    ----------
    Dat: numpy 2D array
        image data
    n: int
    an: list
        angles in degrees
        e.g. [0, 90] or [0, 54.7356, 90] or [0, 45, 90, 135]
    un: list
        order of Legendre polynomials to be used as the expansion
        even polynomials [0, 2, ...] gerade
        odd polynomials [1, 3, ...] ungerade
        all orders [0, 1, 2, ...] 
    inc: int
        number of pixels per Newton sphere

    Returns
    -------
    inv_Data: numpy 2D array
       inverse Abel transformed image
    """

    Dat = np.atleast_2d(Dat)
    rows, cols = Dat.shape
    r2 = rows//2 + rows % 2
    c2 = cols//2 + cols % 2

    #Determine minimum and maximum counts in the image
    min_Dat=np.min(Dat) 
    max_Dat=np.max(Dat)

    centre = np.array([r2, c2])
    span = c2 - 1
    dim = cols
    if verbose:
         print(centre, span, dim, Dat.shape)


    #Number of used polynoms
    pol = len(un)       

    if verbose:
        Summary=['The clipped and rounded VMI has size: {}'.format(Dat.shape),
                 'The chosen set of angles is: {}'.format(an),
                 'The {} chosen Polynoms are : P{}'.format(pol,np.array(un)),
                 'The resolution is: {} pixel'.format(inc)]
        for item in Summary:
            print(item)

    # Calculate Projections at chosen angles.
    proj=len(an)                  #How many projections
 
    QLz =np.zeros((proj, dim))     #Define array for projections.

    # Rotate and project VMI-image for each angle (as many as projections)
    an=np.array(an)
    if an.all == [0, 90]:
    #If coordinates of the detector coincide with the projection directions 
    # unnecessary rotationsare avoided, i.e.an=[0, 90] degrees
        QLz[0]=np.sum(Dat,axis=1)
        QLz[1]=np.sum(Dat,axis=0)
    else:
        for i in range(proj):
            Rot_Dat=sci.ndimage.interpolation.rotate(Dat, an[i], axes=(1, 0),
                                                     reshape=False)
            QLz[i,:]=np.sum(Rot_Dat,axis=1) 

    QLz.shape,np.swapaxes(QLz,0,1).shape

    #Calculation of Base vectors
    #Define triangular matrix containing columns x/y (representing cos(θ)).
    n = cols
    Index = np.indices((n, n))[:, :, :]
    Index[:,0,0] = 1
    cos = Index[0]*np.tri(n, n, k=0, )[::-1, ::-1]/np.diag(Index[0])

    #Concatenate to "bi"-triangular matrix 
    COS = np.concatenate((-cos[::-1, :],cos[1:, :]), axis=0)
    TRI = np.concatenate((np.tri(n, n, k=0,)[:-1, ::-1],
                          np.tri(n, n, k=0,)[::-1, ::-1]), axis=0)

    #inc: use only each inc other vector. Keep the base vector with full span
    COS = COS[:, ::-inc]
    TRI = TRI[:, ::-inc]

    COS = COS[:, ::-1] #rearrange base vectors again in ascending order
    TRI = TRI[:, ::-1]

    #clip=0
    #clip first vectors (smallest Newton spheres) to avoid singularities
    #COS=COS[:,clip:]
    #It is difficult to trace the effect on the SVD solver used below.
    #TRI=TRI[:,clip:] #Usually no clipping works fine.

    #Define Basis vectors for a given polynomial order "order" and a 
    #i given projection angle "angle".
    def bas(ord, angle, COS, TRI):
        """Define Basis vectors for a given polynomial order "order" and a 
           given projection angle "angle".
        
        Parameters
        ----------
        ord: int
            polynomial order
        angle: float
            projection angle
        COS: numpy array
            bi-triangular matrix containing the base for each Newton sphere
        TRI: numpy array
            bi-triangular matrix containing the base for each Newton sphere

        Returns
        -------
        basis_vec: numpy array

        """ 
        basis_vec = sci.special.eval_legendre(ord, angle)*\
                    sci.special.eval_legendre(ord, COS)*TRI
        return basis_vec
    
    #Calculate base vectors for each projection and each order.
    B = np.zeros((pol, proj, len(COS[:, 0]), len(COS[0, :])))
    Norm = np.sum(bas(0, 1, COS, TRI), axis=0)  #calculate normalization
    an_rad = np.radians(an)  #Express angles in radians

    for p in range(pol):
        for u in range(proj):
            B[p, u, :, :] = bas(un[p], np.cos(an_rad[u]), COS, TRI)/Norm 

    #Concatenate vectors to one matrix of bases
    Bpol = np.concatenate((B), axis=2)
    Basis = np.concatenate((Bpol), axis=0)     

    if verbose:
        print('Number of base vectors for each Polynom = ', len(Basis[0,:])/pol)
        comment= 'If you want to check for\"roundness\", you work best with' +\
                 ' two angles.\nE.g.: [0,90] \nIf the widths of the' +\
                 ' projections are quite different\nyou should consider' +\
                 ' to scale the VMI appropriately.' +\
                 ' (See above)\n\nIf the projections are very' +\
                 ' asymmetric you may want to invoke an expansion \nusing' +\
                 ' uneven Legendre polynoms.'
        print (comment)


    # Solve equation system for Newtonsphere radii and their anisotropies
    # This is the heart of the VMI evaluation with Lin_Basex as described in 
    # the paper.
    # The solution of this linear equation yields the beta values of all 
    # involved spheres.
    # All subsequent cell concerns only the representation of the found values.
    # 
    # lstsq solves the equation system invoking an SVD decomposition.
    # Look up the definition of lstsq to learn about the use of rcond and the 
    # data provided.
    # Choose rcond (typically 0.001) big enough that there is a solution with
    # reasonable beta's 
    # i.e., in the order of maximal counts per sphere, but at the same time as 
    # small as possible to avoid averaging.
    # You can increase rcond until you note a “smearing out“ effect.


    def beta_solve(Basis, bb, rcond=0.0005, clip_low=0, clip_high=0,
                   verbose=False):
        # set rcond to zero to switch conditioning off
        #clip the β's for the "clip_low" smallest and "clip_high" biggest 
        #Newton spheres.

        #define array for solutions. len(Basis[0,:])//pol is an integer.
        Beta = np.zeros((pol, len(Basis[0, :])//pol))
        rr = len(Beta[0,:])
        if clip_high == 0: 
            clip_high = rr
        #solve equation
        Sol = np.linalg.lstsq(Basis, bb, rcond)
        #arrange solutions into subarrays for each β.
        Beta = Sol[0].reshape((pol, len(Sol[0])//pol)) 
        #To avoid an error message use integer divison to define the shape
        rr=len(Beta[0,:])

        if verbose:
            print('Sums of residuals (squared Euclidean 2-norm) :',Sol[1])
            print()
            print('Number and dimensions of projections: ',QLz.shape)
            print('Number of used polynoms and number of depicted'
                  ' Newtonspheres: ',Beta[:,clip_low:clip_high].shape)
            print()


        return Beta
    
    #arrange all projections for input into "lstsq"
    bb = np.concatenate(QLz, axis=0)

    Beta = beta_solve(Basis, bb)

    def SL(i, x, y, Beta_convol):
        """Calculates interpolated β(r), where r= radius"""
        r = np.sqrt(x**2 + y**2 + 0.1)  # + 0.1 to avoid divison by zero.
        #normalize:divison by circumference.
        BB = np.interp(r, index, Beta_convol[i, :], left=0)/(2*np.pi*r)
        return BB*eval_legendre(un[i], x/r)

    def Slices(sig_s=0.5):
        """defines sigma for Gaussian smoothing function and 
           calculates Slices"""

        NP = len(Beta[0,:])           #number of points in 3_d plot.
        index=range(NP)

        Beta_convol=np.zeros((pol,NP))
        Slice_3D=np.zeros((pol,2*NP, 2*NP)) 

        #Define smoothing function
        Basis_s = np.fromfunction(
                      lambda i: np.exp(-(i-(NP)/2)**2/(2*sig_s**2))/\
                                        (sig_s*2.5),(NP,))

        #Convolve Beta's with smoothing function
        for i in range(pol):
            Beta_convol[i] = np.convolve(Basis_s, Beta[i,:], mode='same')
    
        for i in range(pol): #Calculate ordered slices:
            Slice_3D[i] = np.fromfunction(
                      lambda k, l: SL(i, (k-NP),(l-NP), Beta_convol), 
                                      (2*NP, 2*NP))
    
        Slice = np.sum(Slice_3D, axis=0) #Sum ordered slices up

        return Slice
        

    inv_Dat = Slices(sig_s)
   
    return inv_Dat
