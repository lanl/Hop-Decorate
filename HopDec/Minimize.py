# -*- coding: utf-8 -*-

"""
Minimisation module.

"""
import sys
import copy
import time
import math

import numpy as np
from scipy import optimize

from .Lammps import *
from .State import readStateLAMMPSData
from .Input import *
from .Utilities import *
from .Vectors import *

################################################################################

def commandLineArgs():
    """
    Parse command line arguments
    
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimise a lattice file.")
    
    parser.add_argument('inputFile', help="The lattice file to be minimised.")
    parser.add_argument('outputFile', help="The file to store the minimised lattice in.")
    parser.add_argument('-d', dest="dumpMin", default=False, action="store_true", help="Dump the minimization")
    
    return parser.parse_args()


################################################################################
def mainCMD(comm):

    # from . import State
    
    # pull command line arguments
    progargs = commandLineArgs()
    
    # read the minimisation parameters  
    params = getParams()
    
    # read lattice and calculate the forces
    state = readStateLAMMPSData(progargs.inputFile)

    # LAMMPS object
    lmp = LammpsInterface(params)

    # Minimize
    main(state, lmp, params, dumpMin = progargs.dumpMin, verbose = True)
    
    # write relaxed state
    state.writeState(progargs.outputFile)
    
    log(__name__, "Minimized state is stored at: "+ progargs.outputFile, 2)

def main(state, lmp : LammpsInterface, params : InputParams, dumpMin = False, verbose = False):

    # Minimize
    lmp.minimize(state, dump = dumpMin, verbose = verbose)

