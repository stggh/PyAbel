from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import abel
from collections import OrderedDict
import platform

# maximum execution time period before method is excluded from timing
maxtime = 5000 # milliseconds

sizes = np.logspace(np.log10(100), np.log10(3000), 10)
n_max_bs   = 10000
n_max_slow = 10000
transform_repeat = 4 # for less than 400

# note matplotlib v2 colors
results = {'basex': ['C0'], 
           'basex_bs': ['C0'],
           'direct_C': ['C8'],
           'direct_Python': ['C1'],
           'fourier_expansion' : ['C4'],
           'fourier_expansion_bs' : ['C4'],
           'hansenlaw': ['k'],
           'linbasex': ['C7'],
           'linbasex_bs': ['C7'],
           'onion_bordas': ['C6'],
           'onion_peeling': ['C2'],
           'onion_peeling_bs': ['C2'],
           'two_point': ['C9'],
           'two_point_bs': ['C9'],
           'three_point': ['C3'],
           'three_point_bs': ['C3']} 

if int(mpl.__version__[0]) < 2:
    for k, v in results.items():
        results[k] = v.strip('C')

results = OrderedDict(sorted(results.items(), key=lambda t: t[0]))
lines = results.copy()

ax = plt.subplot(111)
ax.set_xlabel('Image size (n)')
ax.set_ylabel('Time (ms)')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(90, 5e3)
ax.set_ylim(0.001, 2.5e4)

ax.set_title("PyAbel benchmark Python{} {} {}"
             .format(platform.python_version(), platform.system(),
                     platform.machine()))

plt.ion()
plt.show()


ns = []
first_time = True
print("size(n)  method                iabel(msec)    basis(msec)   fabel(msec)")
for n in sizes:
    n = int( (n//2)*2 + 1 ) 
    print(" {:d}".format(n))
    ns.append(n)
    
    if n>400: 
        transform_repeat = 1
        
    select = [k for k in results.keys() if k[-3:] != '_bs']
    res = abel.benchmark.AbelTiming([n], select=select, n_max_bs=n_max_bs,
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

    # print results
    for k, v in res.iabel.items():
       print("{:10s} {:20s} {:8.2f}".format(' ', k, v[0]), end='')
       if k+'_bs' in res.bs.keys():
           print(" {:12.2f}".format(res.bs[k+'_bs'][0]), end='')
       else:
           print(" {:12s}".format(' '), end='')
       if k in res.fabel.keys():
           print(" {:12.2f}".format(res.fabel[k][0]), end='')
       print()

    # remove methods that take a long time for the given n
    for k, v in res.bs.items():
       if v[0] > maxtime:
           print(" ** removed {:s} from timimg loop".format(k[:-3]))
           del results[k]
           del results[k[:-3]]
    # slow non basis methods
    for k, v in res.iabel.items():
       if v[0] > maxtime and k in results:
           print(" ** removed {:s} from timimg loop".format(k))
           del results[k]
    
    ax.relim()
    ax.autoscale_view(True, True, True)

    plt.draw(); plt.pause(0.001)
    
    first_time = False


print('complete!!')
plt.ioff()
plt.savefig('plot_pyAbel_benchmarks.png', dpi=100)
plt.show()
