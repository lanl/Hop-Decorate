from .Lammps import *
from .Input import *
from .Utilities import *
from .Vectors import *
from .State import readStateLAMMPSData, getStateCanonicalLabel, State

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

    # pull command line arguments
    progargs = commandLineArgs()
    
    # read the minimisation parameters  
    params = getParams()
    
    # read lattice and calculate the forces
    state = readStateLAMMPSData(progargs.inputFile)

    # Minimize
    main(state, params, dump = progargs.dumpMin, verbose = True, comm = comm)
    
    # write relaxed state
    state.writeState(progargs.outputFile)
    
    log(__name__, f'Minimized state is stored at: {progargs.outputFile}', 2)

def main(state : State, params : InputParams, dump = False, verbose = False, comm = None, lmp = None):

    # Minimize
    if not lmp: lmp = LammpsInterface(params, communicator = comm)
    move = lmp.minimize(state, dump = dump, verbose = verbose)
    
    # labelling
    getStateCanonicalLabel(state, params, comm = comm, lmp = lmp)

    return move