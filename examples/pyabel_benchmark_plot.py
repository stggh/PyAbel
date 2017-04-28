from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import abel
from collections import OrderedDict
import platform

sizes = np.logspace(np.log10(100), np.log10(2000), 10)
n_max_bs   = 10000
n_max_slow = 10000
transform_repeat = 4 # for less than 400

results = {'basex': ['C0'], 
           'basex_bs': ['C0'],
           'direct_C': ['C8'],
           'direct_Python': ['C1'],
           'fourier_expansion' : ['C4'],
           'hansenlaw': ['k'],
           'linbasex': ['C7'],
           'onion_bordas': ['C6'],
           'onion_peeling': ['C2'],
           'onion_peeling_bs': ['C2'],
           'two_point': ['C9'],
           'two_point_bs': ['C9'],
           'three_point': ['C3'],
           'three_point_bs': ['C3']} 

results = OrderedDict(sorted(results.items(), key=lambda t: t[0]))
lines = results.copy()

ax = plt.subplot(111)
ax.set_xlabel('Image size (n)')
ax.set_ylabel('Time (ms)')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(90, 5e3)
ax.set_ylim(0.01, 5e4)

ax.set_title("PyAbel benchmark Python{} {} {}"
             .format(platform.python_version(), platform.system(),
                     platform.machine()))

plt.ion()
plt.show()


ns = []
first_time = True
for n in sizes:
    n = int( (n//2)*2 + 1 ) 
    print('size: %i'%n)
    ns.append(n)
    
    if n>400: 
        transform_repeat = 1
        
    res = abel.benchmark.AbelTiming([n], n_max_bs=n_max_bs,
               n_max_slow=n_max_slow, transform_repeat=transform_repeat)

    for name in results.keys():
        if '_bs' in name:
            results[name].append(res.bs[name])
        else:
            results[name].append(res.iabel[name])

        if first_time:
            if 'bs' in name: 
                ls = 'dotted' 
                marker = '+'
            else:
                ls = 'solid'
                marker = 'o'
            lines[name], = ax.plot(ns, results[name][1:], ls=ls,
                           label=name, marker=marker, color=results[name][0])
            ax.legend(fontsize='small', frameon=False, loc=4, numpoints=1,
                      labelspacing=0.1)
        else:
            lines[name].set_data( ns, results[name][1:])
    
    ax.relim()
    ax.autoscale_view(True, True, True)

    plt.draw(); plt.pause(0.001)
    
    first_time = False


print('complete!!')
plt.ioff()
plt.savefig('PyAbel-benchmarks.png',dpi=100)
plt.show()
