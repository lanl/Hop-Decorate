import os
import sys
import random
import copy
import math
import collections

import numpy as np

from . import Graphs
from . import State
from .Plots import *
from .Input import *
from .Vectors import *
from .State import *
from .Utilities import *
from .Constants import boltzmann

class Transition:
    """
    Stores information about a single transition from a given state to another
    """
    def __init__(self, initialState : State, finalState : State, index=None):

        self.initialState = initialState
        self.finalState = finalState
        self.saddleState = None

        self.initialHash = initialState.canLabel
        self.finalHash = finalState.canLabel

        self.transitionHash = None
        
        self.redecorated = 0
        
        self.forwardBarrier = 0
        self.reverseBarrier = 0
        self.dE = 0
        self.KRA = 0

        self.pathRelativeEnergies = []
        self.imagePositions = []

        self.hash = None
        self.index = index

    def calcRate(self, temperature : float, prefactor = 1e14) -> float:
        return prefactor * np.exp( -self.forwardBarrier/ (temperature * boltzmann) )
    
    def plot(self, folder : str, filename : str):
        
        """
        Plot the NEB energy pathway.

        Args:
            name (str or bool): The filename for the plotted energy pathway.

        """

        f = f'{folder}/NEB_{filename}.png'

        # log(__name__,f"Saving NEB Energy Pathway to '{f}'")

        linePlot(f'{f}',
                 xvals = [ i for i in range(len(self.pathRelativeEnergies)) ],
                 xlabel = 'NEB Node',
                 yvals = self.pathRelativeEnergies,
                 ylabel = 'Relative Energy (eV)')

    def exportStructure(self, folder : str, subfolder : str) -> None:

        """
        Export the NEB structures to a specified folder.

        Args:
            folder (str): The folder where the NEB structures will be exported.

        """
        folder = str(folder)
        subfolder = str(subfolder)
        if not os.path.exists(folder): 
            os.mkdir(folder)
        if not os.path.exists(f'{folder}/{subfolder}'):
            os.mkdir(f'{folder}/{subfolder}')

        # log(__name__,f"Saving NEB Structure to '{folder}/{subfolder}'")

        for s,struc in enumerate(self.imagePositions):
            tempState = copy.copy(self.initialState)
            tempState.pos = struc
            tempState.writeState(f'{folder}/{subfolder}/{s}.dat')
            tempState = None

    def printTransitionSummary(self):
        """
        Display the results of the energy barriers and related values.

        This function prints the forward energy barriers, reverse energy barriers, KRA (average of forward and reverse barriers),
        and dE values in electron volts (eV).

        Attributes:
        self (object): The instance of the class.

        Returns:
        None

        """
    
        log(__name__,f'Results:')
        print(f'''\t    Forward Energy Barrier: {self.forwardBarrier} eV, Rate at 1000 K: {self.calcRate(1000, self.forwardBarrier):e} 1/s
            Reverse Energy Barrer: {self.reverseBarrier} eV, Rate at 1000 K: {self.calcRate(1000, self.reverseBarrier):e} 1/s
            KRA: {self.KRA} eV
            dE: {self.dE} eV ''')

class Connection:

    """ A connection is a collection of transitions which connect two or more states"""
    
    def __init__(self, initialState : State, finalState : State):
        
        self.initialState = initialState
        self.finalState = finalState
        self.saddleState = None
        
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def printResults(self):
        """
        Display the results of the energy barriers and related values.

        This function prints the forward energy barriers, reverse energy barriers, KRA (average of forward and reverse barriers),
        and dE values in electron volts (eV).

        Attributes:
        self (object): The instance of the class.

        Returns:
        None

        """
        
        fBarString = rBarString = kraString = dEString = ''

        for i in range(len(self.transitions)):
            if i > 0:
                fBarString +=  ", "
                rBarString +=  ", "
                kraString +=  ", "
                dEString +=  ", "

            fBarString +=  f"{self.transitions[i].forwardBarrier}"
            rBarString += f"{self.transitions[i].reverseBarrier}"
            kraString += f"{self.transitions[i].KRA}"
            dEString += f"{self.transitions[i].dE}"
        
        log(__name__,f'Results:')
        print(f'''\t    Forward Energy Barrier(s): {fBarString} eV
            Reverse Energy Barrer(s): {rBarString} eV
            KRA(s): {kraString} eV
            dE(s): {dEString} eV ''')
        
    def plot(self, folder = '.'):
        for t,trans in enumerate(self.transitions):
            trans.plot(folder, t)
    
    def exportStructures(self, folder = '.'):
        for t,trans in enumerate(self.transitions):
            trans.exportStructure(folder,t)
            
            

class ASETransition:
    """
    Stores information about a single transition from a given state to another
    """
    def __init__(self, initialState: Atoms, finalState: Atoms):

        self.initialState = initialState
        self.finalState = finalState
        
        self.forwardBarrier = 0
        self.reverseBarrier = 0
        self.dE = 0
        self.KRA = 0

        self.initialState_energy = 0.0
        self.finalState_energy = 0.0

class ASEConnection:

    """ A connection is a collection of transitions which connect two or more states"""
    
    def __init__(self, initialState : Atoms, finalState : Atoms):
        
        self.initialState = initialState
        self.finalState = finalState
        
        self.transitions = []     


def transitionToASE(transition: Transition, params: InputParams):

    aseTrans = ASETransition(transition.initialState.toASE(params), transition.finalState.toASE(params))

    aseTrans.forwardBarrier = transition.forwardBarrier
    aseTrans.reverseBarrier = transition.reverseBarrier
    aseTrans.KRA= transition.KRA
    aseTrans.dE = transition.dE

    aseTrans.initialState_energy = transition.initialState.totalEnergy
    aseTrans.finalState_energy = transition.finalState.totalEnergy

    return aseTrans

def connectionToASE(connection: Connection, params: InputParams):

    aseConn = ASEConnection(connection.initialState.toASE(params), connection.finalState.toASE(params))

    aseConn.transitions = [ transitionToASE(tran, params) for tran in connection.transitions ]
    
    return aseConn
    
if __name__ == '__main__':
    pass