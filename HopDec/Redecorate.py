from .Model import *
from .Input import *
from .Utilities import *
from .State import *
from .NEB import *
from .Plots import *
from .Vectors import *
from . import NEB as nb

import numpy as np
import pandas as pd
import random
import pickle
import copy

class Redecorate:

    def __init__(self, params: InputParams):

        self.params = params
        self.outPath = ''

        self.connections = []
        self.aseConnections = []
        self.transHash = None

    def __len__(self):
        return len(self.aseConnections)

    def buildShuffleLists(self, state):

        initialTypeList = [-1 if ty not in self.params.staticSpeciesTypes else ty for ty in state.type]
        count = initialTypeList.count(-1)
    
        # now we build the list which we want to shuffle.
        nAtEach = count * np.array(self.params.concentration)

        shuffleList = [self.params.activeSpeciesTypes[n] 
                       for n, nAt in enumerate(nAtEach) 
                       for _ in range(int(nAt))
                       ]
        
        while len(shuffleList) < count:
            shuffleList.append(self.params.activeSpeciesTypes[0]) # just puts more of the first element in...

        return initialTypeList, shuffleList

    def run(self, initialState: State, finalState: State, comm = None):
        
        rank = 0
        size = 1
        startI = 0
        endI = self.params.nDecorations
        newComm = None

        if comm:
            
            rank = comm.Get_rank()
            size = comm.Get_size()
            nDecTot = self.params.nDecorations
            eachRank = nDecTot // size

            startI = rank * eachRank
            endI = startI + eachRank

            newComm = comm.Split(color = rank, key = rank)

        if rank == 0: log(__name__, f"Starting NEB Redecoration Campaign")

        initialTypeList, shuffleList = self.buildShuffleLists(initialState)

        seed = self.params.randomSeed
        
        for n in range(startI,endI):

            shuffleListCurrent = copy.deepcopy(shuffleList)
            init = copy.deepcopy(initialState)
            fin = copy.deepcopy(finalState)
            initL = copy.deepcopy(initialTypeList)

            log(__name__, f"rank: {rank}: Redecoration: {n+1}",1)        

            # randomize atom type list
            random.seed(seed * (n + 1))
            random.shuffle(shuffleListCurrent)
            
            # recombine with static types
            j = 0
            for i in range(len(initL)):
                if initL[i] == -1:
                    initL[i] = shuffleListCurrent[j]
                    j += 1

            # apply the atom type list to initial and final states.
            init.type = initL
            fin.type = initL
            init.NSpecies = self.params.NSpecies
            fin.NSpecies = self.params.NSpecies

            # run a NEB
            connection = nb.main(init, fin, self.params, comm = newComm)

            if len(connection):
                self.connections.append(connection)
                
            init = None
            fin = None
            initL = None
            connection = None

        # need to gather with ase structure objects.
        # gathering with State objects caused Seg Fault         
        self.aseConnections = [ connectionToASE(conn, self.params) for conn in self.connections ]
        if comm:           
            connectionList = comm.gather(self.aseConnections, root = 0)
            if rank == 0:
                self.aseConnections = [item for sublist in connectionList for item in sublist]

        
            comm.barrier()
            
        return 0
            
    def pickleIt(self, filename = 'test'):
        
        # Write data to pandas
        df = pd.DataFrame(columns = ['initialState', 
                                     'finalState', 
                                     'KRA', 
                                     'dE', 
                                     'initialState_Energy',
                                     'finalState_Energy',
                                     ])

        # changes structure to ase
        for d,decoration in enumerate(self.aseConnections):

            for t,transition in enumerate(decoration.transitions):

                row = pd.DataFrame.from_dict({'initialState'        : [transition.initialState],
                                              'finalState'          : [transition.finalState],
                                              'KRA'                 : [transition.KRA],
                                              'dE'                  : [transition.dE],
                                              'initialState_Energy' : [transition.initialState_energy], 
                                              'finalState_Energy'   : [transition.finalState_energy],
                                              })
                
                df = pd.concat([df,row], ignore_index=True)

        # Pickle the DataFrame
        with open(f'{filename}', 'wb') as f:

            pickle.dump(df, f)
            
            
    def summarize(self):
        """
        Iterates through the connections and transitions within each decoration, printing information about each transition.

        Args:
            self (object): The instance of the class containing the connections and transitions.

        Returns:
            None: This function doesn't return anything; it only prints information about the connections and transitions.
        """

        log(__name__,"Summary:")

        for r,decoration in enumerate(self.connections):
            print(f'\tConnection {r+1}:')

            for t,transition in enumerate(decoration.transitions):

                print(f'\t\tTransition {t+1}:')
                print(f'\t\t\t{transition.forwardBarrier = }')
                print(f'\t\t\t{transition.dE = }')

################################################################################

def commandLineArgs():
    """
    Parse command line arguments

    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the NEB method.")
    parser.add_argument('initialFile', help="The initial state file.")
    parser.add_argument('finalFile', help="The final state file.")
    parser.add_argument('-i', dest="paramFile", default="HopDec-config.xml", help="Parameter input file (default=HopDec-config.xml)")
    parser.add_argument('-p', dest="plotPathway", nargs='?', const=True, default=False, help="Plot the energy profile with an optional filename")
    parser.add_argument('-v', dest="verbose", default=False, action="store_true", help="print NEB log to screen")
    
    return parser.parse_args()

################################################################################

def mainCMD(comm):

    # get command line arguments
    progargs = commandLineArgs()

    # parameters from the config file
    params = getParams()

    # initial state object
    initialState = readStateLAMMPSData(progargs.initialFile)
    finalState = readStateLAMMPSData(progargs.finalFile)

    transition = Transition(initialState, finalState)

    Red = main(transition, params , comm = comm, pickle = True)

    Red.summarize()

def main(obj, params : InputParams, pickle = True, comm = None):
    
    rank = 0
    if comm: rank = comm.Get_rank()

    # Redecorate a Transition
    if isinstance(obj, Transition):

        # instantiate redecoration results
        Red = Redecorate(params)
    
        # run the redecoration method
        Red.run(obj.initialState, obj.finalState, comm = comm)
        obj.redecorated = 1
        
        # pickle it?
        if pickle and not rank: 
            obj.label(params)
            Red.pickleIt(filename = f'{obj.canLabel}_{obj.nonCanLabel}.pkl')
    
    else:
        raise TypeError("obj must be an instance of Transition")

    if comm: comm.barrier()

    return Red