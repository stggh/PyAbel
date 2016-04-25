# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import matplotlib.pylab as plt

# linbasex - evaluate 1D projections of VM-images in terms of 1D projections of spherical functions,

# load image as a numpy array
# use scipy.misc.imread(filename) to load image formats (.png, .jpg, etc)
print("LB: loading 'data/O2-ANU1024.txt.bz2'")
IM = np.loadtxt("data/O2-ANU1024.txt.bz2")

# inverse Abel transform
AIM = abel.Transform(IM, method='linbasex', center='convolution',
                     center_options=dict(square=True, odd_size=True),
                     transform_options=dict(return_Beta=True))

recon = AIM.transform
speed = AIM.linbasex_angular_integration

# Set up some axes
fig = plt.figure(figsize=(15, 4))
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1))
ax3 = plt.subplot2grid((1, 3), (0, 2))

# raw image
im1 = ax1.imshow(IM, aspect='auto')
fig.colorbar(im1, ax=ax1, fraction=.1, shrink=0.9, pad=0.03)
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
ax1.set_title('velocity map image: size {:d}x{:d}'.format(*IM.shape))

# 2D transform
cols = recon.shape[1]
c2 = cols//2   # half-image width
im2 = ax2.imshow(recon, aspect='auto', vmin=0, vmax=recon[:c2-50, :c2-50].max())
fig.colorbar(im2, ax=ax2, fraction=.1, shrink=0.9, pad=0.03)
ax2.set_xlabel('x (pixels)')
ax2.set_ylabel('y (pixels)')
ax2.set_title('linbasex inverse Abel: size {:d}x{:d}'.format(*recon.shape))

# 1D speed distribution
ax3.plot(speed/speed[200:].max())
ax3.axis(xmax=500, ymin=-0.05, ymax=1.1)
ax3.set_xlabel('speed (pixel)')
ax3.set_ylabel('intensity')
ax3.set_title('speed distribution (Beta[0])')

# Prettify the plot a little bit:
plt.subplots_adjust(left=0.06, bottom=0.17, right=0.95, top=0.89, wspace=0.35,
                    hspace=0.37)

# save copy of the plot
plt.savefig("example_linbasex.png", dpi=100)

plt.show()
