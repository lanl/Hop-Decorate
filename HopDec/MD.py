# -*- coding: utf-8 -*-

"""
MD module.

"""
import sys
import copy
import time
import math

import numpy as np
from scipy import optimize

from .Lammps import *
from .State import *    
from .Input import *
from .Utilities import *
from .Vectors import *
from .Minimize import *

################################################################################

def commandLineArgs():
    """
    Parse command line arguments
    
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimise a lattice file.")
    
    parser.add_argument('inputFile', help="The lattice file to be minimised.")
    # parser.add_argument('outputFile', help="The file to store the minimised lattice in.")
    # parser.add_argument('-d', dest="dumpMin", default=False, action="store_true", help="Dump the minimization")
    
    return parser.parse_args()


################################################################################
def main(state : State, params : InputParams, dump = None, init = True, comm = None, lmp = None):
    """
    Just a driver for MD with a given input file.
    """
    flag = 0
    stateTemp = copy.deepcopy(state)
    if not lmp: lmp = LammpsInterface(params, communicator = comm)

    maxMove = lmp.runMD(stateTemp, params.segmentLength, T = params.MDTemperature, dump=dump)
    if maxMove > params.eventDisplacement:
        flag = 1
        if params.verbose: log(__name__, f'Transition detected in state: {state.canLabel}')
        lmp.minimize(stateTemp, verbose = False)
    
    return lmp, stateTemp, flag
    
