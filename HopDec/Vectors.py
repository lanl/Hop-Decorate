import numpy as np
import random
import math
# from .State import State 
import numpy as np
import random

def separation(pos1 : list, pos2 : list, cellDims : list, doSqrt = True):
    """
    Calculate the Euclidean distance between two positions, considering periodic boundary conditions.

    Parameters:
    - pos1 (array-like): The coordinates of the first position.
    - pos2 (array-like): The coordinates of the second position.
    - cellDims (array-like): The dimensions of the simulation cell.
    - doSqrt (bool, optional): Whether to return the square of the distance or its square root. Default is True.

    Returns:
    - float: The separation distance between pos1 and pos2.

    """
    lengths = [cellDims[0], cellDims[4], cellDims[8]]
    sum_sq = 0
    for i in range(3):
        d = np.abs(pos1[i] - pos2[i])
        if d > lengths[i] / 2:
            d = lengths[i] - d
        sum_sq += d ** 2 
    if doSqrt:
        return np.sqrt(sum_sq)
    else: 
        return sum_sq

def calcForceInfNorm(force : list):
    """
    Calculate the infinity norm of a force array.

    Parameters:
    - force (array-like): The force array.

    Returns:
    - float: The maximum absolute value of the elements in the force array.
    - int: The index of the element with the maximum absolute value in the force array.

    """
    max_force = max(abs(f) for f in force)
    max_index = force.index(max_force)
    return max_force, max_index

def randomVector(N : int, randomSeed=None):
    """
    Generate a normalized vector of length N containing random numbers between -1 and 1.

    Parameters:
    - N (int): The length of the vector.
    - randomSeed (int, optional): Seed for the random number generator. Default is None.

    Returns:
    - array-like: A normalized vector of length N containing random numbers.

    """ 
    if randomSeed:
        random.seed(int(randomSeed))
    
    randVec = np.random.uniform(-1, 1, N)
    return normalise(randVec)

def normalise(vect : list) -> list:
    """
    Normalize a vector.

    Parameters:
    - vect (array-like): The vector to be normalized.

    Returns:
    - array-like: The normalized vector.

    """
    mag = magnitude(vect)
    if mag == 0:
        print("WARNING: attempting to normalize zero vector")
        return vect
    return vect / mag

def magnitude(vect : list) -> float:
    """
    Calculate the magnitude of a vector.

    Parameters:
    - vect (array-like): The vector.

    Returns:
    - float: The magnitude of the vector.

    """
    return np.linalg.norm(np.array(vect))

def displacement(v1, v2, cellDims):
    """
    Calculate the displacement vector between two positions, considering periodic boundary conditions.

    Parameters:
    - v1 (array-like): The coordinates of the first position.
    - v2 (array-like): The coordinates of the second position.
    - cellDims (array-like): The dimensions of the simulation cell.

    Returns:
    - array-like: The displacement vector from v1 to v2.

    """
    lengths = [cellDims[0], cellDims[4], cellDims[8]]
    
    dir = np.zeros(3)
    for i in range(3):
        dir[i] = v1[i] - v2[i]
        if dir[i] > lengths[i] / 2:
            dir[i] = lengths[i] - dir[i]

    return dir

def maxMoveAtom(s1, s2) -> float:
    """
    Find the maximum displacement of an atom between two states.

    Parameters:
    - s1 (State): The initial state.
    - s2 (State): The final state.

    Returns:
    - float: The maximum displacement of an atom between s1 and s2.

    """
    maxMove = 0
    for i in range(s1.NAtoms):
        maxMove = np.max([maxMove, separation(s1.pos[3*i:3*(i+1)], s2.pos[3*i:3*(i+1)], s1.cellDims)])
    return maxMove

def maxMoveAtomPos(pos1 : list, pos2 :  list, cellDims: list) -> float:
    """
    Find the maximum displacement between two sets of atomic positions.

    Parameters:
    - pos1 (array-like): The initial atomic positions.
    - pos2 (array-like): The final atomic positions.
    - cellDims (array-like): The dimensions of the simulation cell.

    Returns:
    - float: The maximum displacement between pos1 and pos2.

    """
    maxMove = 0
    for i in range(len(pos1) // 3):
        maxMove = np.max([maxMove, separation(pos1[3*i:3*(i+1)], pos2[3*i:3*(i+1)], cellDims)])
    return maxMove

def COM(pos, cellDims) -> list:
    """
    Calculate the center of mass of a system of particles within a periodic box.

    Parameters:
    - pos (array-like): The atomic positions.
    - cellDims (array-like): The dimensions of the simulation cell.

    Returns:
    - list: The coordinates of the center of mass.

    """
    coordinates = [pos[3*p:3*(p+1)] for p in range(len(pos) // 3)]
    coordinates = np.array(coordinates)
    box_size = np.array([cellDims[0], cellDims[4], cellDims[8]])

    # Calculate the center of mass within the periodic box
    center_of_mass = np.zeros(3)
    total_mass = 0.0

    for coord in coordinates:
        displacement = coord - np.floor(coord / box_size) * box_size
        distance = np.linalg.norm(displacement)
        center_of_mass += coord / distance
        total_mass += 1.0 / distance

    center_of_mass /= total_mass

    return center_of_mass.tolist()

def findConnectivity(pos, cutoff, cellDims):
    """
    Find connected points within a distance cutoff in a periodic system.

    Parameters:
    - pos (array-like): The atomic positions.
    - cutoff (float): The distance cutoff.
    - cellDims (array-like): The dimensions of the simulation cell.

    Returns:
    - list: A list of connected point pairs.

    """
    points = [pos[3*p:3*(p+1)] for p in range(len(pos) // 3)]
    connected_points = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if separation(points[i], points[j], cellDims) <= cutoff:
                connected_points.append([i, j])
    
    return connected_points
