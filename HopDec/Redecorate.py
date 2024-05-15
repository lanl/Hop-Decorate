import ase
import numpy as np
import tempfile
import random
import pandas as pd
import os
import pickle as pick
import copy

from ase.io import read, write
from ase.calculators.lammpslib import LAMMPSlib
from ase.optimize import MDMin
from ase.neb import NEB

from .Model import *
from . import NEB as nb
from .Input import *
from .Utilities import *
from .State import *
from .NEB import *
from .Plots import *
from .Vectors import *

class Redecorate:

    def __init__(self, params: InputParams):

        self.params = params
        self.outPath = ''

        self.connections = []
        self.aseConnections = []
        self.transHash = None

    def __len__(self):
        return len(self.connections)

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

            if rank == size - 1:
                endI = nDecTot

            newComm = comm.Split(color = rank, key = rank)

        if rank == 0:
            log(__name__, f"Starting NEB Redecoration Campaign")

        initialTypeList = copy.deepcopy(initialState.type)

        count = 0
        for t,ty in enumerate(initialState.type):
            if ty not in self.params.staticSpeciesTypes:
                count += 1
                initialTypeList[t] = -1 # This is a marker to say where we have taken from in our type list
                
        # now we build the list which we want to shuffle.
        nAtEach = count * np.array(self.params.concentration)

        shuffleList = []
        for n,nAt in enumerate(nAtEach):
            for i in range(int(nAt)):
                shuffleList.append(self.params.activeSpeciesTypes[n])

        while len(shuffleList) < count:
            shuffleList.append(random.choice(self.params.activeSpeciesTypes))

        
        for n in range(startI,endI):

            shuffleListCurrent = copy.deepcopy(shuffleList)
            init = copy.deepcopy(initialState)
            fin = copy.deepcopy(finalState)
            initL = copy.deepcopy(initialTypeList)

            log(__name__, f"rank: {rank}: Redecoration: {n+1}",1)        

            # randomize atom type list
            seed = self.params.randomSeed
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
            comm.barrier()            
            connectionList = comm.gather(self.aseConnections, root = 0)
            if rank == 0:
                self.aseConnections = [item for sublist in connectionList for item in sublist]
            comm.barrier()

        return 0
            
    def pickleIt(self, filename = 'test'):
        
        # Write data to pandas
        df = pd.DataFrame(columns = ['initialState', 'finalState', 'KRA', 'dE', 'initialState_Energy','finalState_Energy'])

        # changes structure to ase
        for d,decoration in enumerate(self.aseConnections):
            for t,transition in enumerate(decoration.transitions):
                row = pd.DataFrame.from_dict({'initialState' : [transition.initialState], 'finalState' : [transition.finalState], 'KRA' : [transition.KRA], 'dE' : [transition.dE], 'initialState_Energy': [transition.initialState_energy], 'finalState_Energy': [transition.finalState_energy]})
                df = pd.concat([df,row], ignore_index=True)

        # Pickle the DataFrame
        with open(f'{filename}.pickle', 'wb') as f:
            pick.dump(df, f)
            
            
    def summarize(self):

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

    Red = main(initialState, finalState, params , comm = comm, pickle = True)

    Red.summarize()

def main(initState : State, finState : State, params : InputParams, pickle = True, comm = None):
    
    rank = 0
    if comm:
        rank = comm.Get_rank()
     
    # instantiate HopDecResults
    Red = Redecorate(params)
    
    # run the redecoration method
    Red.run(initState, finState, comm = comm)

    # pickle it?
    if pickle and not rank: 
        Red.pickleIt(filename = getTransitionHash(params, initState, finState))

    if comm: comm.barrier()

    return Red