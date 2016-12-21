#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import time
import warnings

from .transform import *
from .tools import *

class Image(Transform):
    """ Image container class for storing and transforming an image.

    This class holds a single image that may then be manipulated using the
    various Abel package methods. It conveniently provides all available 
    transform and tools methods via the <tab> query. 

    Attributes
    ----------
    Those of the Transform class and tools methods.
    """

    def __init__(self, IM=None, method='hansenlaw', direction='inverse',
                 symmetry_axis=(0, 1), use_quadrants=(True, True, True, True),
                 symmetrize_method='average', angular_integration=False,
                 transform_options=dict(), center_options=dict(),
                 angular_integration_options=dict(),
                 recast_as_float64=True, verbose=False):

        if isinstance(IM, str):
            self.load_image(IM)
            
        Transform.__init__(IM, method=method, direction=direction,
                           symmetry_axis=symmetry_axis,
                           use_quadrants=use_quadrants,
                           angular_integration=angular_integration,
                           transform_options=transform_options,
                           center_options=center_options,
                           angular_integration_options=angular_integration_options,
                           recast_as_float64=recast_as_float64)


    def load_image(self, fn):
        self.IM = np.loadtxt(fn)

    # overloaded to make parameters visible
    def center_image(self, center='com', odd_size=True, square=False,
                     verbose=False):
        self.IM = tools.center.center_image(self.IM, center=center,
                                            odd_size=odd_size, square=square,
                                            verbose=verbose)

    
        
