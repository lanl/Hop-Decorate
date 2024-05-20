
from .Graphs import *
from .Input import *
from .Transitions import *
from .ASE import *
from .Utilities import *
from . import Vectors
from . import Constants
from . import Atoms
from .Lammps import LammpsInterface

from ase import Atoms
from ase.io import read
from ase.io.lammpsdata import read_lammps_data
from scipy.spatial.distance import cdist

import sys
import pickle
import numpy as np
from typing import List

################################################################################

class State:
    def __init__(self, NAtoms):

        """
        Initializes a State object.

        Args:
            NAtoms (int): The number of atoms in the state.

        Attributes:
            NAtoms (int): The number of atoms in the state.
            cellDims (np.ndarray): An array storing the dimensions of the cell.
            pos (np.ndarray): An array for storing the atomic positions.
            type (np.ndarray): An array for storing the types of atoms.
            NSpecies (int): The number of species.
            totalEnergy (float): The total energy of the system.
            time (int): The time associated with the state.
            centro_sym (list): List for storing centro_symmetry values.
            defectCOM (list): List storing defect center of mass.
            defectIndices (list): List of indices of defects.
            defectPositions (list): List of defect positions.
            canLabel (str): Canonical label for the state.
            displacementVector (None or np.ndarray): Vector for displacement.
            displacementNVector (None or np.ndarray): Normalized displacement vector.
            KE (np.ndarray): Kinetic energy array for each atom.
            PE (np.ndarray): Potential energy array for each atom.
        """
        
        self.NAtoms = NAtoms

        # initialise arrays for storing State data
        self.cellDims = np.zeros(9, np.float64)
        self.cellDims[0] = 100.0
        self.cellDims[4] = 100.0
        self.cellDims[8] = 100.0
        self.pos = np.empty(3*self.NAtoms, np.float64)
        self.type = np.empty(self.NAtoms, np.int32)
        self.NSpecies = None        

        self.totalEnergy = None
        self.time = 0
        self.doWork = 0
        self.depth = 0

        self.centroSyms = []
        self.defectCOM = []
        self.defectIndices = []
        self.defectPositions = []
        self.defectTypes = []

        self.canLabel = ''
        self.nonCanLabel = ''

        self.displacementVector = None
        self.displacementNVector = None
        self.index = None

        self.KE = np.zeros(self.NAtoms, np.float64)
        self.PE = np.zeros(self.NAtoms, np.float64)

    def getIndicesFromPositions(self, inputPos, maxSep=0.1):

        """
        Finds the atoms within the State that correspond to the input positions.

        Parameters
        ----------
        inputPos : np.ndarray
            The positions that you want to associate with atoms in the `State`.
        maxSep : float, optional
            The maximum separation allowed for a site to be associated with one of the
            input positions. Defaults to 0.1.

        Returns
        -------
        indices : np.ndarray
            The indices of the atoms that correspond to the `inputPos` positions. Note,
            if no atom was found to correspond to a position then we return -1 in its
            place. It is up to the user what they do with this information.

        Raises
        ------
        ValueError
            If `maxSep` is zero or negative.

        """
        # check maxSep parameter
        if maxSep <= 0:
            raise ValueError("Max separation cannot be zero or negative")

        if not len(inputPos):
            # if no input positions are specified return an empty array
            indices= np.empty(0, np.int32)

        else:
            # create array for storing the result
            indices = np.empty(int(len(inputPos) / 3), np.int32)

            # cell dims array
            cellDims = self.cellDims

            # find indices
            indices = np.array([])
            for inPos in inputPos:
                for index in range(self.NAtoms):
                    pos = [ self.pos[3*index], self.pos[3*index+1], self.pos[3*index+2] ]
                    if Vectors.separation(pos,inPos,cellDims) < maxSep:
                        indices = np.append(indices,index)
                
        return indices

    def setCellDims(self, dimsArray):
        """
        Set the `State` cell dimensions.

        Parameters
        ----------
        dimsArray : array_like[9]
            Array-like object of length 9, containing the dimensions
            you want to set. Does nothing if this array is the
            wrong length.

        """
        if len(dimsArray) != len(self.cellDims):
            return

        self.cellDims[:] = dimsArray[:]
    
    def calcTemperature(self):
        """
        Calculate the temperature of the `State`.

        Parameters
        ----------
        NMoving : int, optional
            The number of moving atoms in the `State` (defaults to all atoms).

        Returns
        -------
        temperature : float
            The temperature of the `State` in K.

        Notes
        -----
        This assumes that all the atoms in the system are 'moving'. If you have
        fixed atoms then this will give a different result to the MD code.

        """

        keSum = np.sum(self.KE)
        temperature = 2.0 * keSum / (3.0 * Constants.boltzmann * self.NAtoms)

        return temperature

    def getMinSepList(self, inputPos, atomList, maxSpacing = 10.0, atomListSize=None):
        """
        Returns minimum separation between `inputPos` and an atom from `atomList`.

        Parameters
        ----------
        inputPos : np.ndarray[3]
            The position you want to check.
        atomList : np.ndarray
            The indices of the atoms from the `State` you want to check against.
        maxSpacing : float
            The maximum spacing between atoms in the State. This is used for domain
            decomposition; if set too small could result in there being no points near
            `inputPos`. You can check if this has happened since `minSep` will be larger
            than `maxSpacing`.
        atomListSize : int, optional
            The size of `atomList`. If this parameter is not set then we use the full
            array (i.e. this parameter can be used if wish to only consider the first
            `atomListSize` elements ofe the `atomList` array).

        Returns
        -------
        minSep : float or None
            The minimum separation found between `inputPos` and an atom from `atomList`.
            If no atom was found within `maxSpacing` of the `inputPos` then retun None.
        minSepIndex : int or None
            The index of the site within `atomList` that was closest to `inputPos` or
            None if no atom was found within `maxSpacing`. Note this is not the index
            within the State but is the index within `atomList`, i.e.
            `0 <= index < len(atomList)`.

        Raises
        ------
        ValueError
            If atomListSize is specified and it is out of range (bigger than
            `len(atomList)` or negative).

        """
        # check atomListSize
        atomListLen = len(atomList)
        if atomListSize is None:
            atomListSize = atomListLen
        elif atomListSize > atomListLen or atomListSize < 0:
            raise ValueError(f"atomListSize out of range ({atomListSize}, {atomListLen})")

        if atomListSize == 0:
            minSep = None
            minSepIndex = None
        else:
            # cell dims array
            cellDims3 = np.empty(3, np.float64)
            cellDims3[0] = self.cellDims[0]
            cellDims3[1] = self.cellDims[4]
            cellDims3[2] = self.cellDims[8]
            
            minSep = maxSpacing
            for i,atomid in enumerate(atomList):
                test_pos = [ self.pos[3*atomid], self.pos[3*atomid+1], self.pos[3*atomid+2] ]
                
                sep = separation(inputPos,test_pos,self.cellDims)
                if  sep < minSep:
                    minSep = sep
                    minSepIndex = i

            # check if we found an atom within maxSpacing of inputPos
            if minSep == maxSpacing:
                minSep = None
                minSepIndex = None

        return minSep, minSepIndex

    def atomSeparation(self, index1, index2):
        """
        Calculate the separation between two atoms.

        Parameters
        ----------
        index1, index2 : integer
            Indexes of the atoms you want to calculate the separation
            between.

        Returns
        -------
        atomSeparation : float
            The separation between the two atoms. This function will
            return 'None' if the indexes are out of range.

        Raises
        ------
        IndexError
            If the specified indexes are too large.

        """
        if index1 < self.NAtoms and index2 < self.NAtoms:
            pass
            atomSeparation = Vectors.separation(self.atomPos(index1), self.atomPos(index2), self.cellDims)
        else:
            raise IndexError(f"Atom index(es) out of range: ({index1} or {index2}) >= {self.NAtoms}")

        return atomSeparation

    def atomPos(self, index : int):
        """
        Return the position of the given atom.

        Parameters
        ----------
        index : integer
            The index of the atom whose position you require.

        Returns
        -------
        atomPos : ndarray
            The position of the selected atom.

        Raises
        ------
        IndexError
            If the specified index is too large.

        Notes
        -----
        The returned array is a pointer to the atoms position in the
        `State.pos` array. Modifying the position in the returned
        array will modify the atoms position in the `State`.

        """
        if index < self.NAtoms:
            atomPos = self.pos[3*index:3*index+3]
        else:
            raise IndexError(f"Atom index out of range: {index} >= {self.NAtoms}")

        return atomPos
    
    def volume(self):
        """
        Compute the volume of the `State`.

        Returns
        -------
            volume : float
                The volume of the `State`.

        """
        return self.cellDims[0] * self.cellDims[4] * self.cellDims[8]
    
    def writeState(self, filename : str) -> None:

        """
        Write a LAMMPS data file containing atomic positions and box dimensions.

        Args:
            filename (str): The name of the file to write the LAMMPS data.

        Returns:
            None
        """

        writeLAMMPSDataFile(filename, self.NAtoms, self.NSpecies, self.cellDims, self.type, self.pos)

    def toASE(self, params : InputParams) -> Atoms:    
        
        """
        Convert the state object to an ASE (Atomic Simulation Environment) object.

        Args:
            params: Input parameters for the conversion.

        Returns:
            Atoms: ASE Atoms object converted from the state object.
        """

        positions = [ ( self.pos[3*i], self.pos[3*i + 1], self.pos[3*i + 2] ) for i in range(self.NAtoms) ]
        
        atomTypeSymbols = []
        for i in range(self.NAtoms):
            atomTypeSymbols.append(params.specieNames[ self.type[i] - 1 ])

        atoms = Atoms(symbols = atomTypeSymbols, 
                      positions = positions, 
                      cell = [self.cellDims[0],self.cellDims[4],self.cellDims[8]],
                      pbc = [1,1,1])

        return atoms
    
    
def readStateLAMMPSData(filename: str)-> State:

    try:
        f = open(filename,"r")
    except:
        sys.exit(f"{__name__}: ERROR: cannot open file: {filename}")

    lines = f.readlines()
    
    # pop comment line
    _ = lines.pop(0)

    # NAtoms line
    line = lines.pop(0)
    if line.split()[-1] != 'atoms':
        sys.exit(f"{__name__}: ERROR: Incorrect file format: line 2 != x atoms")
    else:
        state = State(int(line.split()[0]))

    # atom type lines
    line = lines.pop(0)
    if line.split()[-1] != 'types':
        sys.exit(f"{__name__}: ERROR: Incorrect file format: line 3 != x atom types")
    else:
        state.NSpecies = int(line.split()[0])

    # cell dimensions lines
    # x
    line = lines.pop(0)
    if line.split()[-1] != 'xhi' and line.split()[-2] != 'xlo':
        sys.exit(f"{__name__}: ERROR: Incorrect file format: line 4 != x x xlo xhi")
    elif float(line.split()[0]) != 0.0:
        sys.exit(f"{__name__}: ERROR: Cell must have [0.0,0.0,0.0] as the origin.")
    else:
        state.cellDims[0] = float(line.split()[1])

    # y
    line = lines.pop(0)
    if line.split()[-1] != 'yhi' and line.split()[-2] != 'ylo':
        sys.exit(f"{__name__}: ERROR: Incorrect file format: line 5 != y y ylo yhi")
    elif float(line.split()[0]) != 0.0:
        sys.exit(f"{__name__}: ERROR: Cell must have [0.0,0.0,0.0] as the origin.")
    else:
        state.cellDims[4] = float(line.split()[1])

    # z
    line = lines.pop(0)
    if line.split()[-1] != 'zhi' and line.split()[-2] != 'zlo':
        sys.exit(f"{__name__}: ERROR: Incorrect file format: line 6 != z z zlo zhi")
    elif float(line.split()[0]) != 0.0:
        sys.exit(f"{__name__}: ERROR: Cell must have [0.0,0.0,0.0] as the origin.")
    else:
        state.cellDims[8] = float(line.split()[1])

    # Atoms ...
    line = lines.pop(0)
    line = lines.pop(0)
    line = lines.pop(0)

    # atom list
    typ = []
    pos = []
    for i in range(state.NAtoms):
        line = lines.pop(0)
        
        cols = len(line.split())

        if cols > 5:
            sys.exit(f"{__name__}: ERROR: Incorrect file format: ID TYPE X Y Z only")
        if line.split() == '':
            sys.exit(f"{__name__}: ERROR: Incorrect file format: Missing atom line")

        t = int(line.split()[1])
        if t > state.NSpecies:
            sys.exit(f"{__name__}: ERROR: type {t} > {state.NSpecies} atom types")
        else:
            typ.append(t)
        

        p = [line.split()[2], line.split()[3], line.split()[4]]
        for po in p:
            pos.append(float(po))
    
    state.pos = np.array(pos)
    state.type = np.array(typ)

    return state

def readStateLAMMPSDump(filename: str, params : InputParams)-> List[State]:

    ase = read(filename, index = ':', format = 'lammps-dump-text')
    states = []
    
    for i in ase:

        NAtoms = int(len(i.symbols))
        state = State(NAtoms)

        state.pos = i.positions.flatten()
        state.cellDims = [i.cell[0][0], 0, 0, 0, i.cell[1][1], 0, 0, 0, i.cell[2][2] ]
    
        typeSymbols = params.specieNames
        types = []
        seen = set()
        syms = [x for x in i.symbols if not (x in seen or seen.add(x))]
        for j in range(NAtoms):
            types.append(syms.index(i.symbols[j]) + 1)
        state.type = types
        state.NSpecies = len(typeSymbols)

        states.append(state)

    return states

def asePickleToStateList(params : InputParams, filename : str) -> List[State]:
    """ function to take Matts list of ASE Atoms objects in a pickle and returns a list of States for the a NEB """
    
    nebListState = []
    ase = ASE(params)

    with open(filename, 'rb') as file:
        nebListASE = pickle.load(file)

    for i,atoms in enumerate(nebListASE):
        nebListState.append([ ase.toState(atoms[0]), ase.toState(atoms[1]) ])

    return nebListState


def getStateCanonicalLabel(state : State, params : InputParams, comm = None, lmp = None):

    if not lmp:
        lmp = LammpsInterface(params, communicator = comm)
    lmp.calcCentro(state)

    indices = np.where( state.centroSyms > params.centroCutoff)
    indices = np.array(indices) + 1
    state.defectIndices = indices[0]

    if not len(state.defectIndices):
        print("WARNING: No defect detected. Ensure you minimize before attempting to hash defect")

    defectPos = []
    for index in state.defectIndices:
        i = index - 1
        defectPos.append(state.pos[3*i])
        defectPos.append(state.pos[3*i + 1])
        defectPos.append(state.pos[3*i + 2])
    state.defectPositions = np.array(defectPos)

    # COM
    state.defectCOM = Vectors.COM(state.defectPositions, state.cellDims)
    
    # defect Types
    defectTypes = [ state.type[i-1] for i in indices[0] ]

    state.defectTypes = defectTypes

    graphEdges = Vectors.findConnectivity(state.defectPositions, params.bondCutoff, state.cellDims)
    state.graphEdges = graphEdges
    state.nDefects = nDefectVolumes(graphEdges)

    state.canLabel = graphLabel(graphEdges, types = defectTypes, canonical = 1)
    state.nonCanLabel = graphLabel(graphEdges, indices = state.defectIndices, canonical = 0)


if __name__ == '__main__':
    pass