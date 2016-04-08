# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import numpy as np
import abel

def get_bs_cached(method, cols, basis_dir='.', direction="inverse",
                  verbose=False):
    """load basis set from disk, generate and store if not available.

    Parameters
    ----------
    method : str
        Abel transform method, currently ``two_point`` or ``onion_peeling``
    cols : int
        width of image
    basis_dir : str
        path to the directory for saving / loading the basis
    verbose: boolean
        print information for debugging 

    Returns
    -------
    D: numpy 2D array of shape (cols, cols)
       basis operator array
    """

    basis_generator = {
        "two_point": abel.dasch._bs_two_point,
        "three_point": abel.dasch._bs_three_point,
        "onion_peeling": abel.dasch._bs_onion_peeling,
        "hansenlaw": abel.hansenlaw._bs_hansenlaw,
    }

    if method not in basis_generator.keys():
        raise ValueError("basis generating function for method '{}' not know"
                         .format(method))

    basis_name = "{}_{}basis_{}_{}".format(method, 
                   direction+'_' if direction=="forward" else '', cols, cols)

    D = None
    if basis_dir is not None:
        path_to_basis_file = os.path.join(basis_dir, basis_name)
        if os.path.exists(path_to_basis_file):
            if verbose:
                print("Loading {} operator matrix...".method)
            try:
                D = np.load(path_to_basis_file)
            except ValueError:
                raise
            except:
                raise
    if D is None:
        if verbose:
            print("A suitable operator matrix for '{}' was not found.\n"
                  .format(method), "A new operator matrix will be generated.")
            if basis_dir is not None:
                print("But don\'t worry, it will be saved to disk \
                    for future use.\n")
            else:
                pass

        D = basis_generator[method](cols, direction)

        if basis_dir is not None:
            np.savez(path_to_basis_file, D)
            if verbose:
                print("Operator matrix saved for later use to,")
                print(' '*10 + '{}'.format(path_to_basis_file))
    return D
