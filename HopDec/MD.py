from .Lammps import *
from .State import *    
from .Input import *
from .Utilities import *
from .Vectors import *
from . import Minimize

import copy
import numpy as np

################################################################################
def main(state : State, params : InputParams, dump = None, comm = None, lmp = None, maxMDTime = np.inf, rank = 0, T = None, segmentLength = None, verbose = False):
    
    """
    Use MD to find a hop.
    """

    time = 0
    flag = 0
    stateTemp = copy.deepcopy(state)
    
    if not segmentLength: segmentLength = params.segmentLength
    if not T: T = params.MDTemperature
    if not lmp: lmp = LammpsInterface(params, communicator = comm)

    while not flag and time < maxMDTime:
        
        if verbose: log(__name__, f'rank {rank}: Running MD in state: {state.canLabel} - {state.nonCanLabel}',1)
        maxMove = lmp.runMD(stateTemp, segmentLength, T = T, dump = dump)
        
        if maxMove > params.eventDisplacement:

            flag = 1
            if verbose: log(__name__, f'rank {rank}: Transition detected in state: {state.canLabel} - {state.nonCanLabel}')
            Minimize.main(stateTemp, params, verbose = False, lmp = lmp)

        time += params.segmentLength
        state.time += params.segmentLength

    return lmp, stateTemp, flag
    
