import numpy as np
import random

def displacement(v1, v2, cellDims):
    """
    Calculate the displacement vector between two positions, considering periodic boundary conditions.

    Parameters:
    - v1 (array-like): The coordinates of the first position (shape: (3,)).
    - v2 (array-like): The coordinates of the second position (shape: (3,)).
    - cellDims (array-like): The dimensions of the simulation cell (shape: (9,)).

    Returns:
    - np.ndarray: The displacement vector from v1 to v2 (shape: (3,)).

    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    lengths = np.array([cellDims[0], cellDims[4], cellDims[8]])
    
    displacement = v2 - v1
    displacement = displacement - np.round(displacement / lengths) * lengths
    
    return displacement

def distance(pos1 : list, pos2 : list, cellDims : list):
    """
    Calculate the Euclidean distance between two positions, considering periodic boundary conditions.

    Parameters:
    - pos1 (array-like): The coordinates of the first position.
    - pos2 (array-like): The coordinates of the second position.
    - cellDims (array-like): The dimensions of the simulation cell.
    - doSqrt (bool, optional): Whether to return the square of the distance or its square root. Default is True.

    Returns:
    - float: The distance between pos1 and pos2.

    """
    return magnitude(displacement(pos1,pos2,cellDims))

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
    return vect / magnitude(vect)

def magnitude(vect : list) -> float:
    """
    Calculate the magnitude of a vector.

    Parameters:
    - vect (array-like): The vector.

    Returns:
    - float: The magnitude of the vector.

    """
    return np.linalg.norm(np.array(vect))

def maxMoveAtom(s1, s2) -> float:
    """
    Find the maximum displacement of an atom between two states.

    Parameters:
    - s1 (State): The initial state, with `s1.pos` (array of atomic positions) and `s1.cellDims` (cell dimensions).
    - s2 (State): The final state, with `s2.pos` (array of atomic positions).

    Returns:
    - float: The maximum displacement of an atom between s1 and s2.

    """

    # Reshape positions into (NAtoms, 3) arrays for vectorized operations
    pos1 = np.array(s1.pos).reshape((-1, 3))
    pos2 = np.array(s2.pos).reshape((-1, 3))
    
    # Calculate displacements for all atoms at once
    displacements = pos2 - pos1
    
    # Apply periodic boundary conditions using the optimized displacement logic
    cell_lengths = np.array([s1.cellDims[0], s1.cellDims[4], s1.cellDims[8]])
    displacements = displacements - np.round(displacements / cell_lengths) * cell_lengths
    
    # Calculate the Euclidean norm (magnitude) of displacements for each atom
    distances = np.linalg.norm(displacements, axis=1)
    
    # Return the maximum displacement
    return np.max(distances), np.argmax(distances)

def maxMoveAtomPos(pos1: list, pos2: list, cellDims: list) -> float:
    """
    Find the maximum displacement between two sets of atomic positions using vectorized NumPy operations.

    Parameters:
    - pos1 (list or array-like): The initial atomic positions (flattened [x1, y1, z1, x2, y2, z2, ...]).
    - pos2 (list or array-like): The final atomic positions (same format as pos1).
    - cellDims (list or array-like): The dimensions of the simulation cell (3x3 matrix or [Lx, 0, 0, 0, Ly, 0, 0, 0, Lz]).

    Returns:
    - float: The maximum displacement between pos1 and pos2.

    """
    # Step 1: Reshape positions into (n_atoms, 3) arrays
    pos1 = np.array(pos1).reshape((-1, 3))
    pos2 = np.array(pos2).reshape((-1, 3))
    
    # Step 2: Extract the lengths of the cell (assuming 3x3 cellDims)
    box_lengths = np.array([cellDims[0], cellDims[4], cellDims[8]])
    
    # Step 3: Calculate displacement vector with periodic boundary conditions
    diff = pos2 - pos1
    diff -= np.round(diff / box_lengths) * box_lengths  # Minimum image convention
    
    # Step 4: Compute the Euclidean distance for each atom
    distances = np.linalg.norm(diff, axis=1)
    
    # Step 5: Return the maximum distance
    max_move = np.max(distances)
    
    return max_move

def COM(points, cellDims):
    """
    Calculate the center of mass (COM) of a set of points in a periodic box using the minimum image convention.
    
    Parameters:
    - points: A 1D array containing the 3D coordinates of all points, i.e., [x1, y1, z1, x2, y2, z2, ...]
    - cellDims: A 9-element list or 3x3 matrix defining the periodic box dimensions. It is assumed to be an orthogonal box.
    
    Returns:
    - com: The 3D coordinates of the center of mass.
    """
    # Reshape points into an (n_points, 3) array
    points = points.reshape((-1, 3))
    
    # Extract the lengths of the cell from cellDims (assuming it is a 3x3 matrix)
    box_lengths = np.array([cellDims[0], cellDims[4], cellDims[8]])  # Extract [Lx, Ly, Lz]
    
    # Reference position (can use the position of the first point as a reference)
    ref_point = points[0]
    
    # Apply the minimum image convention relative to the reference point
    for i in range(3):  # Loop over x, y, z
        delta = points[:, i] - ref_point[i]  # Difference from the reference position
        points[:, i] = ref_point[i] + delta - box_lengths[i] * np.round(delta / box_lengths[i])  # Apply periodic boundary
    
    # Calculate the center of mass (mean position of the points)
    com = np.mean(points, axis=0)

    # put COM back inside box
    com = com % box_lengths
    
    return com

def findConnectivity(pos, cutoff, cellDims):
    """
    Find connected points within a distance cutoff in a periodic system.

    Parameters:
    - pos (array-like): The atomic positions, a flat list or 1D array of size 3N, 
      where N is the number of atoms (3 coordinates per atom).
    - cutoff (float): The distance cutoff.
    - cellDims (array-like): The dimensions of the simulation cell (can be a 3x3 or 1x3 array).

    Returns:
    - list: A list of connected point pairs.
    """
    
    # Extract the periodic box dimensions from the 3x3 cellDims matrix if needed
    box_dims = np.array([cellDims[0], cellDims[4], cellDims[8]])
    
    # Reshape positions to an (N, 3) array
    points = np.array(pos).reshape(-1, 3)
    
    # Number of points
    N = len(points)
    
    # Compute distance between each pair of points using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    # Apply periodic boundary conditions
    diff -= np.round(diff / box_dims) * box_dims
    
    # Compute squared distances
    dist_squared = np.sum(diff**2, axis=2)
    
    # Identify which distances are less than the cutoff (squared to avoid sqrt)
    mask = (dist_squared <= cutoff**2) & (dist_squared > 0)  # Ignore self-connections
    
    # Get the indices of the connected points (i, j) pairs
    i, j = np.where(np.triu(mask, k=1))  # Use np.triu to avoid duplicate pairs (i, j) and (j, i)
    
    # Combine the indices into a list of pairs
    connected_points = np.column_stack((i, j)).tolist()
    
    return connected_points