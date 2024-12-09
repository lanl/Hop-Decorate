from .Utilities import *
from .Vectors import *
from .Utilities import log

from lammps import lammps

import random
import ctypes

class LammpsInterface:
    """
    LAMMPS interface object.

    """
    def __init__(self, params, communicator = None):
 
        self._lmp = None
        self._params = params
        self.startPos = None
        self.communicator = communicator
        
    def __del__(self):
        if self._lmp is not None:
            self._lmp.close()

    def _initialiseLammps(self, state):

        """
        Initialise the LAMMPS interface.

        Parameters
        ----------
        obj : `State.State`-like object
            The object that is going to be used with LAMMPS. Must be like the `State.State` object.
        numAtoms : int
            The number of atoms.
        currentPos : `numpy.ndarray`
            The positions of the atoms.
        velocity : `numpy.ndarray`, optional
            If specified then initialise the atoms with the given velocities.
        screenOutput : `True` or `False`, optional
            If False (the default) then disable LAMMPS screen output. This will be overriden by the `debug` argument to
            the initialiser.
        logOutput : `True` or `False`, optional
            If False (the default) then disable LAMMPS log file output.

        Raises
        ------
        LammpsUnitsNotSupported
            If the value of the lammpsUnits parameter is not supported.
        IOError
            If the LAMMPS input file cannot be opened.

        """

        # first close the current instance
        if self._lmp is not None:
            self._lmp.close()

        # create a new instance
        self._lmp = lammps(cmdargs=['-screen','none','-log','none'])#, comm = self.communicator)
        # self._lmp = lammps(cmdargs=['-screen','none'], comm = self.communicator)
        # self._lmp = lammps()

        # initialise
        self._lmp.commands_string(f"""clear
            units metal
            dimension 3
            boundary p p p
            atom_style {self._params.atomStyle}
            atom_modify sort 0 0 # makes PE's come out in right order
            atom_modify map array
            region simBox block 0 {state.cellDims[0]} 0 {state.cellDims[4]} 0 {state.cellDims[8]} units box
            create_box {self._params.NSpecies} simBox
            """)
        
        # initialise the atoms in the box at random positions
        self._lmp.command(f'create_atoms 1 random {state.NAtoms} 8923 NULL')
        
        # define current atom types
        for i in range(state.NAtoms):
            self._lmp.command(f"set atom {i+1} type {state.type[i]}")

        # scatter current atomic positions
        lmp_positions = list(state.pos.ravel())
        c_double_array = (ctypes.c_double * len(lmp_positions))
        lmp_c_positions = c_double_array(*lmp_positions)
        self._lmp.scatter_atoms('x', 1, 3, lmp_c_positions)

        # run the rest of the boilerplate lammps script
        self._lmp.commands_string(self._params.LAMMPSInitScript)
        
        
    def minimize(self, state, initialise = True, etol = None, ftol = None, maxiter = None, dump = False, verbose = True):

        """
        Run a minimisation using LAMMPS on a State in-place

        Parameters
        ----------
        state : `State.State`
            The state object to be minimized.
        initialise : `True` or `False`, optional
            If `True` (default) then initialise LAMMPS before running the minimisation. If the state has not been
            modified since the last time LAMMPS was called then you do not need to initialise.

        """

        if initialise:
            # initialse lammps
            self._initialiseLammps(state)

        # prepare to minimize
        params = self._params
        if ftol is None:
            ftol = params.minimizationForceTolerance
        if etol is None:
            etol = params.minimizationEnergyTolerance
        if maxiter is None:
            maxiter = params.minimizationMaxSteps

        if verbose:
            log(__name__, f"Minimizing. F Tol: {ftol}.",1)

        if dump:
            dumpFn = "dump*.dat"
            dumpFreq = 1
            self._lmp.command(f"dump dumpdat all custom {dumpFreq} {dumpFn} id type x y z")

        self.startPos = np.array(self._lmp.gather_atoms("x", 1, 3))

        
        # run the minimisation
        self._lmp.command(f"minimize {etol} {ftol} {maxiter} {maxiter}")


        # get data back from LAMMPS and store in the state
        maxMove = self.extractData(state) 

        if verbose:
            log(__name__, f"Completed Minimization, E: {round(state.totalEnergy, 4)} eV")

        return maxMove


    def runMD(self, state, runTime, T = None, dump=None, init = True) -> float:
        """
        Run a molecular dynamics simulation.

        """

        params = self._params

        # initialse lammps
        if init:
            self._initialiseLammps(state)

        # finish setting up lammps
        self._lmp.commands_string(f"""
        compute temperature all temp
        """)

        if T is None:
            T = params.MDTemperature

        self._lmp.commands_string(f"""
        velocity all create {0.5*T} {random.randint(0, 999999)}
        fix LANG all langevin {T} {T} 0.1 {random.randint(0, 999999)} gjf no
        fix NVE all nve
        fix COM all recenter INIT INIT INIT
        """)
        
        # thermodynamic info (may change it if not need)
        thermoFreq = 100
        self._lmp.commands_string(f"""
        thermo {thermoFreq}
        thermo_style custom step time c_temperature etotal ke pe
        thermo_modify lost warn
        """)

        # dump file
        if dump:
            dumpFn = "dump*.dat"
            dumpFreq = 1000
            self._lmp.command(f"dump dumpdat all custom {dumpFreq} {dumpFn} id type x y z")

        # set timestep
        dt = self._params.MDTimestep
        self._lmp.command(f"timestep {dt}")

        self.startPos = np.array(self._lmp.gather_atoms("x", 1, 3))
        
        # run simulation
        self._lmp.command(f"run {runTime} post no")

        # get data back from LAMMPS and store in the state
        maxMove = self.extractData(state)

        # update time we ran md in this state.
        state.time += runTime
                
        return maxMove

    def extractData(self, state) -> float:
        """
        Get data back from LAMMPS and store it.

        Parameters
        ----------
        state : `State.State`
            The `State` object that the LAMMPS data will be stored in.

        """
        
        # update positions in state
        state.pos[:] = self._lmp.gather_atoms("x", 1, 3)[:]

        # total energy and force magnitude
        state.totalEnergy = self._lmp.extract_compute("thermo_pe", 0, 0)
        
        # measure the max that an atom has moved
        maxSep = 0
        for i in range(state.NAtoms):
            maxSep = np.max([ maxSep, distance( [ self.startPos[3*i], self.startPos[3*i+1], self.startPos[3*i+2]], [ state.pos[3*i], state.pos[3*i+1], state.pos[3*i+2]] , state.cellDims) ])
        
        return maxSep
    

    def calcCentro(self, state):

        # initialse lammps
        self._initialiseLammps(state)

        self._lmp.command(f'compute CENTRO all centro/atom {self._params.centroN}')

        self._lmp.command('run 0')

        state.centroSyms = self._lmp.numpy.extract_compute("CENTRO",1,1)

if __name__ == "__main__":
    pass