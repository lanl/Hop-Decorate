from .State import *
from .Input import *
from .ASE import *
from .Utilities import *
from .Lammps import *
from .Plots import *
from .Transitions import *
from . import Redecorate as Red

import ase
import numpy as np
import tempfile
import copy

from ase.io import read, write
from ase.calculators.lammpslib import LAMMPSlib
from ase.optimize import MDMin
from ase.neb import NEB
from ase.optimize.fire import FIRE as QuasiNewton
from ase.optimize import BFGS

class ASE_NEB(ASE):
    
    """
    Class representing the Nudged Elastic Band (NEB) method in the ASE package.

    Attributes:
        forwardBarrier (float): The energy barrier from the initial to the saddle point.
        reverseBarrier (float): The energy barrier from the final to the saddle point.
        KRA (float): The average of the forward and reverse barrier energies.
        dE (float): The difference between final and initial energies.
        pathRelativeEnergy (numpy.ndarray): Array of relative energies with respect to the initial energy.
        pathEnergy (list): List of potential energies extracted from the images.
        minimaNodes (list): List of indices corresponding to minima points on the energy path.
        saddleNodes (list): List of indices corresponding to saddle points on the energy path.
        initialCanLabel (None or str): The canonical label for the initial state.
        finalCanLabel (None or str): The canonical label for the final state.
        imagePositions (list): List of the positions of images in the NEB pathway.
        flag (None or int): A flag indicating any issues encountered during the NEB calculation.

    """

    def __init__(self,params : InputParams , communicator = None):
        
        """
        Initialize the ASE_NEB class.

        Args:
            params (InputParams): The parameters for the NEB calculation.
            communicator (None or obj): An optional communicator object.

        """
        
        super().__init__(params, communicator)

        self.forwardBarrier = None
        self.reverseBarrier = None
        self.KRA = None
        self.dE = None
        self.pathRelativeEnergy = None
        self.pathEnergy = None

        self.minimaNodes = []
        self.saddleNodes = []

        self.initialCanLabel = None
        self.finalCanLabel = None
        self.imagePositions = []

        self.badEnds = 0
        self.flag = 0
        
    def run(self, initialState : State, finalState : State, logNEB = None, verbose = False) -> None:
        
        """
        Run the NEB calculation.

        Args:
            initialState (State): The initial state for the NEB calculation.
            finalState (State): The final state for the NEB calculation.
            logNEB (None or str): The log file for NEB results.
            verbose (bool): A flag for verbose output.

        Returns:
            int: Returns 0 after completing the NEB calculation.

        """
        
        # store initial State
        self.initialState = initialState

        if verbose:
            log(__name__,'Initializing ASE Structure')

        # save the canonical label to the NEB object for use by the model class.
        self.initialCanLabel = initialState.canLabel
        self.finalCanLabel = finalState.canLabel

        # transorm out state objects to ase format.
        atomsInitial = initialState.toASE(self.params)
        atomsFinal = finalState.toASE(self.params)

        num_images = self.params.NEBNodes

        images = [atomsInitial] + [atomsInitial for _ in range(num_images-2)] +  [atomsFinal]
        
        neb_images =[]
        for c in images:
            neb_image = c.copy()
            neb_image.calc = self.calculator(3)
            neb_image.pbc = np.array([1, 1, 1], dtype=bool)
            neb_image.wrap()
            neb_images.append(neb_image)

        if verbose:
            log(__name__,f'Interpolating {num_images} images')
        
        # neb with climb = False for stability of path
        band = NEB(neb_images, climb = False, k = self.params.NEBSpringConstant)
        band.interpolate(mic=True)

        if verbose:
            log(__name__,f'''Performing NEB. Force Tol: {self.params.NEBForceTolerance}, Transition:
        {self.initialCanLabel} -> {self.finalCanLabel}''')


        relax = QuasiNewton(band,logfile = logNEB)
        relax.run(fmax = self.params.NEBForceTolerance,
                steps = self.params.NEBMaxIterations)

        # then turn climb on if requested after path is converged
        if self.params.NEBClimbingImage:
            band = NEB(neb_images, climb = self.params.NEBClimbingImage, k = self.params.NEBSpringConstant)
            relax = QuasiNewton(band,logfile = logNEB)
            relax.run(fmax = self.params.NEBForceTolerance, 
                      steps = self.params.NEBMaxIterations)

        
        if verbose:
            log(__name__,f'NEB finished Successfully. Extracting Results')
        self.getResults(neb_images)
        
        # need to pull positions
        self.imagePositions = neb_images

    def getResults(self, images) -> None:
        """
        Update the NEB results in the class attributes based on the provided list of images.

        Args:
            self (obj): The instance of the class.
            images (list): List of images to extract results from.

        Returns:
            None

        Sets the following attributes:
            pathEnergy (list): List of potential energies extracted from the images.
            forwardBarrier (float): Rounded maximum energy minus the initial energy.
            reverseBarrier (float): Rounded maximum energy minus the final energy.
            dE (float): Rounded difference between final and initial energy.
            KRA (float): Rounded average of forward and reverse barrier energies.
            pathRelativeEnergy (numpy.ndarray): Array of relative energies with respect to the initial energy.
        
        """
        
        r = 4 # rounding digit
        e = [config.get_potential_energy() for config in images]
        self.pathEnergy = e

        saddleEnergy = np.max(e)

        self.pathRelativeEnergy = np.array(e) - e[0]
        self.forwardBarrier = round(saddleEnergy - e[0], r)
        self.reverseBarrier = round(saddleEnergy - e[-1], r)
        self.dE = round(e[-1] - e[0],r)
        self.KRA = round(0.5 * (self.forwardBarrier + self.reverseBarrier), r)
        
    
    def findSaddleNodes(self) -> None:
        
        """
        Find the saddle points on the energy path.

        """

        e = self.pathEnergy

        for i in range(1,len(e)-1):
            if e[i] > e[i+1] and e[i] > e[i-1]: 
                    self.saddleNodes.append(i)

    def findMinimaNodes(self) -> None:
        
        """
        Find the minima points on the energy path.

        """

        e = self.pathEnergy

        self.minimaNodes.append(0)
        
        for i in range(1,len(e)-1):
            if e[i] < e[i+1] and e[i] < e[i-1]:
                self.minimaNodes.append(i)

        self.minimaNodes.append(len(e) - 1)

    def checkNEB(self):
        
        """
        Check the NEB path for integrity.

        """

        # check the path for Saddles and Minima
        self.findSaddleNodes()
        self.findMinimaNodes()

        # these indicate something went wrong with the NEB
        if 0 not in self.minimaNodes or len(self.pathEnergy) - 1 not in self.minimaNodes:
            self.badEnds = 1

        if  len(self.saddleNodes) == 0: # or len(self.saddleNodes) != len(self.minimaNodes) - 1:
            self.flag = 1
        


################################################################################

def exportForDebug(init,fin,neb,index):
    
    transition = Transition(init,fin)
    transition.forwardBarrier = neb.forwardBarrier
    transition.reverseBarrier = neb.reverseBarrier
    transition.dE = neb.dE
    transition.KRA = neb.KRA
    
    # also store saddle configuration.
    sad = copy.deepcopy(init)
    sad.pos = neb.imagePositions[ neb.saddleNodes[0] ].get_positions().flatten()
    transition.saddleState = sad

    # save the energies of the images on the path
    transition.pathRelativeEnergies = list(neb.pathRelativeEnergy)

    # save images as states
    transition.imagePositions = [ struc.get_positions().flatten() for struc in neb.imagePositions ]

    # plot and export
    transition.exportStructure('test', index)
    transition.plot('test', index)
    


def commandLineArgs():
    
    """
    Parse command line arguments for running the NEB method.

    Returns:
        obj: The parsed command line arguments.

    """

    import argparse

    parser = argparse.ArgumentParser(description = "Run the NEB method.")
    parser.add_argument('initialFile', help = "The initial state file.")
    parser.add_argument('finalFile', help = "The final state file.")
    parser.add_argument('-i', dest = "paramFile", default = "HopDec-config.xml", help = "Parameter input file (default=HopDec-config.xml)")
    parser.add_argument('-p', dest = "plotPathway", nargs = '?', const = True, default = False, help = "Plot the energy profile with an optional filename")
    parser.add_argument('-l', dest = "logNEB", default = None, action = "store_true", help = "print ASE log of NEB to screen")
    parser.add_argument('-e', dest = "exportStructures", default = None, action = "store_true", help = "export structure files for NEB")
    
    return parser.parse_args()

################################################################################


def mainCMD(comm):

    """
    The main function to run the NEB method.

    Args:
        comm (obj): The communicator object.

    """

    # parameters from the config file
    params = getParams()

    # get command line arguments
    progargs = commandLineArgs()

    # initial state objects
    initialState = readStateLAMMPSData(progargs.initialFile)
    finalState = readStateLAMMPSData(progargs.finalFile)

    # run the NEB main function
    connection = main(initialState, finalState, params, comm, plotPathways = progargs.plotPathway, exportStructures = progargs.exportStructures, verbose = True)

    # print results to the screen
    log(__name__,f"Completed {len(connection.transitions)} NEBs!")



################################################################################


def main(initialState : State, finalState : State, params : InputParams, comm = None, plotPathways = False, exportStructures = False, logNEB = None, verbose = False, pickle = False, directory = './') -> Connection:
    
    """
    The main function to run the NEB calculation.

    Args:
        initialState (State): The initial state for the NEB calculation.
        finalState (State): The final state for the NEB calculation.
        params (InputParams): The parameters for the NEB calculation.
        comm (None or obj): The communicator object.
        plotPathways (bool): A flag indicating whether to plot the energy pathway.
        exportStructures (bool): A flag indicating whether to export the NEB structures.
        logNEB (None or str): The log file for NEB results.
        verbose (bool): A flag for verbose output.
        pickle (bool): A flag indicating whether to pickle the output.

    Returns:
        obj: Returns the connection object containing the NEB results.

    """
    
    maxSaddles = 20
    
    # Store NEB results in a connection and return this object - this is a collection of transitions
    connection = Connection(initialState, finalState)

    # add initial NEB todo:
    nebsTODO = [[initialState,finalState]]
    completedNEBPos = []

    # counters
    nebsCompleted = 0
    nebSuccessful = 0

    # We loop until there are no NEBs left to do or until one of the NEBs fails/ we have done too many
    while len(nebsTODO) and nebsCompleted < params.maxNEBsToDo:

        # set these as the working neb
        workingNEB = nebsTODO.pop(0)
        init = workingNEB[0]
        fin = workingNEB[1]

        # Minimize end points
        if verbose:
            log(__name__,f'Minimizing End Points')
            
        lmp = LammpsInterface(params, communicator = comm)
        minDistInit = lmp.minimize(init, verbose = verbose)
        del lmp

        lmp = LammpsInterface(params, communicator = comm)
        minDistFin = lmp.minimize(fin, verbose = verbose)
        del lmp
        
        # if the minimization took us far from where we started throw it away.
        if minDistInit > params.maxMoveMin or minDistFin > params.maxMoveMin:
            if verbose: log(__name__,f"WARNING: Initial or Final structure moved > {params.maxMoveMin}. Skipping...")
            continue

        if maxMoveAtom(init, fin) < 0.3: # TODO: Make this a parameter
            if verbose: log(__name__,f"WARNING. Initial and Final Structures are the Same. Skipping...")
            continue

        getStateCanonicalLabel(init, params, comm = comm)
        getStateCanonicalLabel(fin, params, comm = comm)

        # NEB object and run
        neb = ASE_NEB(params, communicator = comm)
        
        # Embarrasing Hack. Sometimes ase NEB goes wrong for some reason (?)
        try:
            neb.run(init, fin,
                    logNEB = logNEB, 
                    verbose = verbose)
        except:
            continue

        # check for intermediate minima
        if verbose: log(__name__,f'Checking NEB path for integrity')
        neb.checkNEB()

        # incremenet neb counter
        nebsCompleted += 1

        # check if the NEB did something bad
        if neb.flag:
            print("ERROR: Something went wrong with your NEB")
            continue
        
        if neb.forwardBarrier > params.NEBmaxBarrier:
            print(f"ERROR: Found Barrier {neb.forwardBarrier} > Max Barrier {params.NEBmaxBarrier}")
            continue
        
        if len(neb.saddleNodes) > maxSaddles:
            print(f"Number of Saddles = {len(neb.saddleNodes)} > {maxSaddles}")
            continue

        # If we get this point we know atleast that our NEB is 'fine'
        # It may have multiple minima but we can deal with that
        
        # exportForDebug(init,fin,neb,nebsCompleted)
        
        # Check if we have done this neb before.
        # If not add it to completedNEBs list.
        
        # pull image positions of neb and saddle index
        imagePos = [ x.get_positions().flatten() for x in neb.imagePositions ]
        saddleNode = neb.saddleNodes[0]

        currentNEBPos = [ imagePos[neb.minimaNodes[0]], imagePos[ saddleNode ], imagePos[ neb.minimaNodes[-1] ] ]
        
        seen = 0
        for seenNEBs in completedNEBPos:
            if maxMoveAtomPos(currentNEBPos[0] , seenNEBs[0], init.cellDims) < 0.1 and maxMoveAtomPos(currentNEBPos[1] , seenNEBs[1], init.cellDims ) < 0.1 and maxMoveAtomPos(currentNEBPos[2] , seenNEBs[2], init.cellDims ) < 0.1:
                seen = 1
        if seen: 
            if verbose: print('WARNING: Seen this exact NEB before. Skipping...')
            continue
        else:
            # add it to completedNEBs
            completedNEBPos.append(currentNEBPos)
            
        # Here we want to check if there are intermediate minima evident from the energy profile
        if len(neb.minimaNodes) > 2:
            if verbose: log(__name__, 'Found Intermediate Minima, Requeueing...')
            nebsTODO_temp = []
            for i in range(len(neb.minimaNodes) - 1):
                
                _init = copy.copy(init)
                _fin = copy.copy(fin)

                _init.pos = neb.imagePositions[ neb.minimaNodes[i] ].get_positions().flatten()
                _fin.pos = neb.imagePositions[ neb.minimaNodes[i+1] ].get_positions().flatten()

                # nebsTODO.append([_init, _fin])
                nebsTODO_temp.append([_init, _fin])
            
            # update nebsTODO:
            nebsTODO = nebsTODO_temp + nebsTODO
            continue

        _init = None
        _fin = None

        # We go here if the energy profile looks 'normal' ....
        # If we get here then we know that we havent seen this NEB before...
        requeueFlag = 0
        newMinimaPos = []
        
        # create lists of nodes on the left and right of the saddle with their positions
        lNodes = [ pos for p,pos in enumerate(imagePos) if p < saddleNode and p > 0 ]
        rNodes = [ pos for p,pos in enumerate(imagePos) if p > saddleNode and p < len(imagePos)-1 ]

        # lTest = copy.deepcopy(init)
        # lTest.pos = imagePos[0]
        lTest = init

        # rTest = copy.deepcopy(init)
        # rTest.pos = imagePos[-1]
        rTest = fin
        
        dummyState = copy.copy(init)
        for n,node in reversed(list(enumerate(lNodes))):

            # minimize in-place
            dummyState.pos = node
            lmp = LammpsInterface(params, communicator = comm)
            lmp.minimize(dummyState, verbose = False)
            del lmp

            if maxMoveAtom(lTest, dummyState) < 0.1:
                break
            elif maxMoveAtom(rTest, dummyState) < 0.1:
                continue
            else:
                newMinimaPos.append(node)
                # rTest.pos = node
                requeueFlag = 1
                break

        if len(newMinimaPos): newMinimaPos.reverse()

        if not requeueFlag:
            
            # lTest = copy.deepcopy(init)
            # lTest.pos = imagePos[0]
            lTest = init
            # rTest = copy.deepcopy(init)
            # rTest.pos = imagePos[-1]
            rTest = fin
            # dummyState = copy.deepcopy(init)
            for n,node in enumerate(rNodes):
                
                # minimize in-place
                # dummyState = copy.deepcopy(init)
                dummyState.pos = node
                lmp = LammpsInterface(params, communicator=comm)
                lmp.minimize(dummyState, verbose = False)
                del lmp

                if maxMoveAtom(rTest, dummyState) < 0.1:
                    break
                elif maxMoveAtom(lTest, dummyState) < 0.1:
                    continue
                else:
                    newMinimaPos.append(node)
                    lTest.pos = node
                    requeueFlag = 1
                    break

        newMinimaPos.insert(0,imagePos[0])
        newMinimaPos.append(imagePos[-1])

        if requeueFlag:
            if verbose: log(__name__, 'Found Intermediate Minima, Requeueing...')
            nRequeue = len(newMinimaPos) - 1            
            nebsTODO_temp = []
            for i in range(nRequeue):

                _init = copy.copy(init)
                _fin = copy.copy(fin)

                _init.pos =  newMinimaPos[i]
                _fin.pos = newMinimaPos[i+1]

                # nebsTODO.append([_init, _fin])
                nebsTODO_temp.append([_init, _fin])
                
            # update nebsTODO:
            nebsTODO = nebsTODO_temp + nebsTODO
        _init = None
        _fin = None
        
        if not requeueFlag:
            
            # store the structures and findings in a transition object
            transition = Transition(init,fin)
            transition.forwardBarrier = neb.forwardBarrier
            transition.reverseBarrier = neb.reverseBarrier
            transition.dE = neb.dE
            transition.KRA = neb.KRA
            
            # also store saddle configuration.
            sad = dummyState
            sad.pos = neb.imagePositions[ neb.saddleNodes[0] ].get_positions().flatten()
            transition.saddleState = sad
            
            # save the energies of the images on the path
            transition.pathRelativeEnergies = list(neb.pathRelativeEnergy)
            
            # save images as states
            transition.imagePositions = [ struc.get_positions().flatten() for struc in neb.imagePositions ]

            # plot and export
            if plotPathways: transition.plot(directory, nebSuccessful)
            if exportStructures: transition.exportStructure(directory, nebSuccessful)

            # summarize results
            if verbose: transition.printTransitionSummary()

            # add this transition to the connection object
            connection.transitions.append(transition)

            nebSuccessful += 1
            transition = None

    return connection


if __name__ == '__main__':
    pass