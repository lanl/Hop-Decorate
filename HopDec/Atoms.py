"""
Get information about the elements.

"""
import sys
import os

from . import Constants

_atomicNumberDict = {}
_atomicMassDict = {}
_atomNameDict = {}
_covalentRadiusDict = {}
_RGBDict = {}
_specieDict = {}

def atomicSymbol(amu):
    """
    Returns the symbol of an element given the mass.
    
    Parameters
    ----------
    amu : float
        The mass of the element in question, in AMU.
    
    Returns
    -------
    sym : str[2]
        The symbol of the element with the given mass.
    
    Raises
    ------
    ValueError
        If there is no match for the given mass.
    
    """
    for i in _specieDict:
        if (_atomicMassDict[_specieDict[i]] == amu):
            return _specieDict[i]
            break
    
    raise ValueError("No atomic symbol for %f AMU" % amu)

def atomicNumber(sym):
    """
    Returns atomic number of the element with the given symbol.
    
    Parameters
    ----------
    sym : str[2]
        The symbol of the element.
    
    Returns
    -------
    atomicNumber : int
        Atomic number of the element.
    
    Raises
    ------
    KeyError
        If the given symbol is not recognised.
    
    """
    return _atomicNumberDict[sym]

def atomicMassAMU(sym):
    """
    Returns the mass of the element with the given symbol.
    
    Parameters
    ----------
    sym : str[2]
        The symbol of the element.
    
    Returns
    -------
    atomicMassAMU : float
        Atomic mass of the element in AMU.
    
    Raises
    ------
    KeyError
        If the given symbol is not recognised.
    
    """
    return _atomicMassDict[sym]

def atomicMass(sym):
    """
    Returns the mass of the element with the given symbol.
    
    Parameters
    ----------
    sym : str[2]
        The symbol of the element.
    
    Returns
    -------
    atomicMass : float
        Atomic mass of the element in MD code units.
    
    Raises
    ------
    KeyError
        If the given symbol is not recognised.
    
    Notes
    -----
    The mass is calculated as:
    
    MASS = ATOMIC_MASS(AMU)*AMUTKG*AFSTMS*AFSTMS/EVTJUL,
    
    where:
    
    AMUTKG = 1.672E-27 ; AFSTMS= 1.0E5 ; EVTJUL= 1.6021E-19
    
    """
    return _atomicMassDict[sym] * Constants.massConversion

def atomName(sym):
    """
    Returns the name of the element with the given symbol.
    
    Parameters
    ----------
    sym : str[2]
        The symbol of the element.
    
    Returns
    -------
    atomName : str
        The name of the given element.
    
    Raises
    ------
    KeyError
        If the given symbol is not recognised.
    
    """
    return _atomNameDict[sym]

def covalentRadius(sym):
    """
    Returns the covalent radius of the element with the given symbol.
    
    Parameters
    ----------
    sym : str[2]
        The symbol of the element.
    
    Returns
    -------
    covRad : float
        Covalent radius of the element.
    
    Raises
    ------
    KeyError
        If the given symbol is not recognised.
    
    """
    return _covalentRadiusDict[sym]


def RGB(sym):
    """
    Returns RGB values for the element with the given symbol.
    
    Parameters
    ----------
    sym : str[2]
        The symbol of the element.
    
    Returns
    -------
    rgb : list[3]
        List containing RGB values for the element.
    
    Raises
    ------
    KeyError
        If the given symbol is not recognised.
    
    """
    return _RGBDict[sym]

def initialise():
    """
    Read data in Atoms.IN file into dictionaries.
    
    """
    global _atomicNumberDict, _atomicMassDict, _atomNameDict, _covalentRadiusDict, _RGBDict, _specieDict
    
    path = os.path.join(os.path.dirname(__file__), "data")
    if len(path):
        filename = os.path.join(path, 'Atoms.IN')
    else:
        filename = 'Atoms.IN'
    
    if os.path.exists(filename):
        try:
            f = open(filename, "r")
        except:
            sys.exit('error: could not open atoms file: %s' % filename)
    else:
        sys.exit('error: could not find atoms file: %s' % filename)
    
    # read into dictionaries
    _atomicNumberDict.clear()
    _atomicMassDict.clear()
    _atomNameDict.clear()
    _covalentRadiusDict.clear()
    _RGBDict.clear()
    _specieDict.clear()
    
    count = 0
    for line in f:
        line = line.strip()
        
        array = line.split()
        
        key = array[3]
        # if len(key) == 1:
            # key = key + '_'
        
        _atomicNumberDict[key] = int(array[0])
        _atomicMassDict[key] = float(array[1])
        _atomNameDict[key] = array[2]
        _covalentRadiusDict[key] = float(array[4])
        _RGBDict[key] = [float(array[5]), float(array[6]), float(array[7])]
        _specieDict[count] = key
        
        count += 1
        
    f.close()

if __name__ == '__main__':
    pass
else:
    initialise()
