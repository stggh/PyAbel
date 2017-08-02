# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
from scipy import ndimage
import matplotlib.pylab as plt
import time

IM = np.loadtxt("data/O2-ANU1024.txt.bz2")
zoom = 1
IM = ndimage.zoom(IM, zoom)

IMc = abel.tools.center.center_image(IM, center="com")

Nl = 0
Nu = 50
tf = time.time()
FIM = abel.Transform(IMc, method='fourier_expansion',
                     transform_options=dict(basis_dir="bases", Nl=Nl, Nu=Nu),
                     angular_integration=True)
tf = time.time() - tf

t2 = time.time()
TIM = abel.Transform(IMc, method='two_point',
                     transform_options=dict(basis_dir="bases"),
                     angular_integration=True)
t2 = time.time() - t2

print("fourier in {:g} seconds".format(tf))
print("2pt in {:g} seconds".format(t2))

Trad, TPES = TIM.angular_integration
Frad, FPES = FIM.angular_integration
plt.plot(Trad, TPES/TPES.max(), label="2pt")
plt.plot(Frad, FPES/FPES.max(), label=r"Fourier expansion $Nu={:d}$".format(Nu))
plt.axis(xmin=140*zoom, xmax=440*zoom)
plt.xlabel("radius (pixels)")
plt.ylabel("intensity")
plt.title(r"O$_2{^-}$ photoelectron spectrum")
plt.legend()

plt.savefig("plot_example_fourier_expansion.png", dpi=75)
plt.show()
