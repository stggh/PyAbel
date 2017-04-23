# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import matplotlib.pylab as plt

IM = np.loadtxt("O2-ANU1024.txt.bz2")

AIM = abel.Transform(IM, method='fourier_expansion', center='com')
