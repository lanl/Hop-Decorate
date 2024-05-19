from ase.io import read, write
from ase.io import lammpsdata
import ase.io.lammpsdata

import datetime
import shutil
import versioneer
import numpy as np
from typing import List, Union

def writeTerminalBlock(message : str) -> None:
    
    """
    Writes a message surrounded by '#' characters in a terminal-like block.

    Parameters:
    - message (str): The message to be displayed within the terminal block.

    Returns:
    - None
    
    """

    length = 100
    messagePad = ' ' + message + ' '
    messageLength = len(messagePad)
    if messageLength % 2 != 0:
        messagePad = messagePad + ' '
    
    for _ in range(length): print('#',end='')
    print('')

    for _ in range((length - messageLength) // 2): print('#',end='')
    print(messagePad,end='')
    for _ in range((length - messageLength) // 2): print('#',end='')
    print('')

    for _ in range(length): print('#',end='')
    print('')

def printConsoleHeader():
    # version=versioneer.get_version()
    # writeTerminalBlock(f'Hop-Decorate ({version})')
    writeTerminalBlock(f'Hop-Decorate')

def printConsoleFooter():
    writeTerminalBlock(f'Fin.')

def log(caller: str, message: str, indent: int = 0) -> None:

    """
    Log output to the screen with optional indentation.

    The function prints the log message to the screen, including the current timestamp,
    the caller name, and the message. It also supports an optional indentation level
    specified by the 'indent' parameter.

    Parameters:
        caller (str): The name of the caller or log source.
        message (str): The message to be logged.
        indent (int, optional): The number of indentation levels (defaults to 0).

    Returns:
        None
    """
    # Only print if less than the level set in Input module
    now = datetime.datetime.now().strftime("%d/%m/%y, %H:%M:%S")
    ind = "  " * indent

    try:
        sp = caller.split('.')
        if sp[0] == 'HopDec':
            caller = sp[-1]
    except:
        pass

    print(f"[{now}]: {ind}{caller} >> {message}", flush = True)


def ORlC(filePath: str, popNo: int = 0) -> List[List[str]]:
    """
    Opens a file, reads the lines, pops any from the top that you don't want, and returns those lines as an array.

    The function reads the content of the file specified by 'filePath' and returns its lines as a list of lists.
    Each sublist represents a line from the file, and the resulting list will exclude the first 'popNo' lines if specified.

    Parameters:
        filePath (str): The path to the file that needs to be opened.
        popNo (int, optional): The number of lines to remove from the top. Defaults to 0.

    Returns:
        List[List[str]]: A list of lists containing the lines of the file.
            Each sublist represents a line from the file.

    Raises:
        FileNotFoundError: If the specified file is not found.
        ValueError: If 'popNo' is negative.
    """

    if popNo < 0:
        raise ValueError("The 'popNo' parameter must be a non-negative integer.")

    try:
        with open(filePath, 'r') as f:
            lines = f.readlines()
            return [lines[i:] for i in range(popNo, len(lines))]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: '{filePath}'")

    return lines

def Extract(lst: List[List[Union[int, float, str]]], i: int) -> List[Union[int, float, str]]:
    """
    Extracts the i-th element from each sublist in a list of lists.

    Given a list of lists 'lst' and a non-negative integer 'i', this function extracts
    the i-th element from each sublist and returns a new list containing these elements.

    Parameters:
        lst (List[List[Union[int, float, str]]]): A list of lists (NxM array) where M > 1.
            Each sublist can contain elements of type int, float, or str.
        i (int): The index (i-th element) to extract from each sublist.

    Returns:
        List[Union[int, float, str]]: A new list containing the i-th element from each sublist.
            The order of elements in the returned list corresponds to the order of sublists in 'lst'.

    Raises:
        ValueError: If 'i' is negative or if the input list 'lst' is empty.
    """
    if i < 0:
        raise ValueError("The index 'i' must be a non-negative integer.")

    if not lst:
        raise ValueError("The input list 'lst' must not be empty.")

    return [item[i] for item in lst]

def del_from_arrays(list_of_arrays: List[np.ndarray], index: int) -> List[np.ndarray]:
    """
    Deletes the element at the specified index from each array in the list.

    Parameters:
        list_of_arrays (List[np.ndarray]): A list of NumPy arrays.
        index (int): The index of the element to delete from each array.

    Returns:
        List[np.ndarray]: A new list of NumPy arrays with the specified element removed from each array.

    Raises:
        ValueError: If the index is out of range for any of the arrays.
    """

    # Input validation: Check if the index is within the valid range for the arrays
    array_shapes = [array.shape[0] for array in list_of_arrays]
    if index < 0 or any(index >= shape for shape in array_shapes):
        raise ValueError("Index is out of range for one or more arrays.")

    return [np.delete(array, index, axis=0) for array in list_of_arrays]

def copy_file(source_path: str, destination_path: str):
    """
    Copies a file from the source path to the destination path using the shutil.copy function.

    Parameters:
        source_path (str): The path of the file to be copied.
        destination_path (str): The path where the file will be copied.

    Raises:
        FileNotFoundError: If the source file is not found.
        Exception: If any other unexpected error occurs during copying.

    Returns:
        int: Always returns 0 after the copy operation is completed or an error is encountered.
    """
    try:
        shutil.copy(source_path, destination_path)
        print(f"File '{source_path}' successfully copied to '{destination_path}'.")
    except FileNotFoundError:
        print(f"Error: File '{source_path}' not found.")
    except Exception as e:
        print(f"An error occurred while copying the file: {e}")


def writeLAMMPSDataFile(filename : str, NAtoms : int, NSpecies: int, cellDims : List[float], types : List[int], positions : List[float]):
        """
        Write LAMMPS data file.

        This function writes a LAMMPS data file with the provided parameters, which can be used as input for LAMMPS simulations.

        Parameters:
            filename (str): The name of the output file to be created.
            NAtoms (int): The total number of atoms in the system.
            NSpecies (int): The number of unique atom types/species in the system.
            cellDims (list): A 1D list of 9 elements representing the cell dimensions for the simulation box.
                            The elements are ordered as [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz].
            types (list): A 1D list of length NAtoms containing the type/species of each atom in the system.
                        The type values should range from 1 to NSpecies.
            positions (list): A 1D list of length 3*NAtoms containing the atomic positions in the system.
                            The positions should be ordered as [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN].

        Returns:
            None

        Notes:
            - This function will create a new file or overwrite the existing file with the provided 'filename'.
            - The function writes the header information for the LAMMPS data file, including atom count, cell dimensions,
            and atomic positions.

        """
        
        f = open(filename,'w+')
        f.write('# \n')
        f.write(f'{NAtoms} atoms \n')
        f.write(f'{NSpecies} atom types \n')
        f.write(f'0.0 {cellDims[0]} xlo xhi \n')
        f.write(f'0.0 {cellDims[4]} ylo yhi \n')
        f.write(f'0.0 {cellDims[8]} zlo zhi \n')
        f.write(f'\n')
        f.write(f'Atoms # atomic\n')
        f.write(f'\n')
        for i in range(NAtoms):
            f.write(f'{i+1} {types[i]} {positions[3*i]} {positions[3*i+1]} {positions[3*i+2]} \n')
        f.close()
