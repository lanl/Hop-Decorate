import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import random
import pickle
import copy

# Assuming the Redecorate class and all dependencies are in the same directory or package
from HopDec.Redecorate import Redecorate 
from HopDec.Input import InputParams 
from HopDec import NEB
from HopDec import State

class TestRedecorate(unittest.TestCase):

    def setUp(self):
        """Set up any variables or objects required for testing."""
        # Create a mock InputParams object
        self.params = MagicMock()
        self.params.staticSpeciesTypes = [1, 2]
        self.params.activeSpeciesTypes = [3, 4]
        self.params.concentration = [0.5, 0.5]
        self.params.nDecorations = 2
        self.params.randomSeed = 42
        self.params.NSpecies = 5
        
        # Create a mock State object
        self.initialState = MagicMock()
        self.initialState.type = [1, 3, 3, 2, 4]
        
        self.finalState = MagicMock()
        self.finalState.type = [2, 4, 3, 1, 3]
        
        # Create an instance of Redecorate
        self.redecorate = Redecorate(self.params)

    def test_init(self):
        """Test the __init__ method."""
        self.assertEqual(self.redecorate.params, self.params)
        self.assertEqual(self.redecorate.connections, [])
        self.assertEqual(self.redecorate.aseConnections, [])
    
    def test_len(self):
        """Test the __len__ method."""
        self.redecorate.aseConnections = [2, 3, 4]
        self.assertEqual(len(self.redecorate), 3)
    
    def test_buildShuffleLists(self):
        """Test the buildShuffleLists method."""
        state = MagicMock()
        state.type = [1, 3, 3, 2, 4]
        
        initialTypeList, shuffleList = self.redecorate.buildShuffleLists(state)
        
        # Check that types not in staticSpeciesTypes are set to -1
        self.assertEqual(initialTypeList, [1, -1, -1, 2, -1])
        
        # Check the shuffle list has the expected number of elements
        self.assertEqual(len(shuffleList), 3)  # 3 active atoms in state


    
    @patch('HopDec.Utilities.log')
    def test_summarize(self, mock_log):
        """Test the summarize method."""
        
        # Create mock connection data
        mock_transition = MagicMock(forwardBarrier=0.5, dE=0.3)
        mock_decoration = MagicMock(transitions=[mock_transition])
        self.redecorate.connections = [mock_decoration]
        
        with patch('builtins.print') as mock_print:
            self.redecorate.summarize()
            
            # Check that the summary information was printed
            mock_print.assert_any_call('\tConnection 1:')
            mock_print.assert_any_call('\t\tTransition 1:')
            mock_print.assert_any_call(f'\t\t\ttransition.forwardBarrier = 0.5')
            mock_print.assert_any_call(f'\t\t\ttransition.dE = 0.3')

if __name__ == '__main__':
    unittest.main()