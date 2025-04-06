import unittest
import numpy as np
from HopDec.Vectors import *

class TestVectors(unittest.TestCase):
    
    ''' Vectors.distance '''
    def test_distance_without_periodic(self):
        pos1 = [1.0, 2.0, 3.0]
        pos2 = [4.0, 5.0, 6.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        self.assertAlmostEqual(distance(pos1, pos2, cellDims), np.sqrt(27), places=5)
        
    def test_distance_with_periodic_1(self):
        pos1 = [1.0, 2.0, 3.0]
        pos2 = [9.0, 9.0, 9.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]

        self.assertAlmostEqual(distance(pos1, pos2, cellDims), np.sqrt(29), places=5)

    def test_distance_with_periodic_2(self):
        pos1 = [1.0, 1.0, 1.0]
        pos2 = [3.0, 9.0, 9.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]

        self.assertAlmostEqual(distance(pos1, pos2, cellDims), np.sqrt(12), places=5)

    def test_distance_with_periodic_edge(self):
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [10.0, 10.0, 10.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]

        self.assertAlmostEqual(distance(pos1, pos2, cellDims), 0, places=5)

    ''' Vectors.randomVector '''
    def test_randomVector(self):
        N = 3
        vec = randomVector(N, randomSeed=42)
        self.assertEqual(len(vec), N)
        mag = magnitude(vec)
        self.assertAlmostEqual(mag, 1, places=5)  # Vector should be normalized
    
    ''' Vectors.normalise '''
    def test_normalise(self):
        vect = [3, 4, 0]
        norm_vect = normalise(vect)
        self.assertAlmostEqual(magnitude(norm_vect), 1, places=5)
    
    def test_normalise_zero_vector(self):
        vect = [0, 0, 0]
        norm_vect = normalise(vect)
        self.assertEqual(vect, norm_vect)
    
    ''' Vectors.magnitude '''
    def test_magnitude(self):
        vect = [3, 4, 0]
        self.assertEqual(magnitude(vect), 5)
    
    ''' Vectors.displacement '''
    def test_displacement(self):
        v1 = [1.0, 2.0, 3.0]
        v2 = [9.0, 9.0, 9.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        displacement_vec = displacement(v1, v2, cellDims)
        np.testing.assert_almost_equal(displacement_vec, [-2.0, -3.0, -4.0], decimal=5)
    
    def test_maxMoveAtom(self):
        class MockState:
            def __init__(self, pos, cellDims, NAtoms):
                self.pos = pos
                self.cellDims = cellDims
                self.NAtoms = NAtoms
        
        pos1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        pos2 = [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        s1 = MockState(pos1, cellDims, 2)
        s2 = MockState(pos2, cellDims, 2)
        
        max_move = maxMoveAtom(s1, s2)[0]
        self.assertAlmostEqual(max_move, np.sqrt(27), places=5)
    
    def test_maxMoveAtomPos(self):
        pos1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        pos2 = [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        max_move = maxMoveAtomPos(pos1, pos2, cellDims)
        self.assertAlmostEqual(max_move, np.sqrt(27), places=5)
    
    def test_COM_periodic(self):
        points = np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        com = COM(points, cellDims)
        np.testing.assert_almost_equal(com, [0.0, 0.0, 0.0], decimal=5)

    def test_COM_realistic(self):
        points = np.array([1.53e-15, 12.6525, 16.2675, 1.66e-15, 10.845, 18.075, 1.57e-15, 12.6525, 19.8825, 1.79e-15, 14.46, 18.075, 18.075, 12.6525, 16.2675, 19.8825, 10.845, 16.2675, 18.075, 10.845, 18.075, 18.075, 12.6525, 19.8825, 19.8825, 10.845, 19.8825, 19.8825, 14.46, 16.2675])
        cellDims = [21.69, 0, 0, 0, 21.69, 0, 0, 0, 21.69]
        
        com = COM(points, cellDims)
        np.testing.assert_almost_equal(com, [20.06325, 12.291  , 17.89425], decimal=5)
    
    def test_findConnectivity(self):
        pos = [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 8.0, 8.0, 8.0]
        cutoff = 1.0
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        connected_points = findConnectivity(pos, cutoff, cellDims)
        expected_connections = [[0, 1]]
        self.assertEqual(connected_points, expected_connections)
    
    def test_findConnectivity_periodic(self):
        pos = [1.0, 2.0, 3.0, 9.5, 2.0, 3.0]  # Close when considering periodic boundaries
        cutoff = 2.0
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        connected_points = findConnectivity(pos, cutoff, cellDims)
        expected_connections = [[0, 1]]  # These points should be connected
        self.assertEqual(connected_points, expected_connections)

    def test_findConnectivity_realistic(self):
        pos = [1.54e-15, 9.0375, 16.2675, 1.78e-15, 7.23, 18.075, 1.58e-15, 9.0375, 19.8825, 1.66e-15, 10.845, 18.075, 18.075, 9.0375, 16.2675, 19.8825, 7.23, 16.2675, 18.075, 7.23, 18.075, 18.075, 9.0375, 19.8825, 19.8825, 7.23, 19.8825, 19.8825, 10.845, 16.2675, 18.075, 10.845, 18.075, 19.8825, 10.845, 19.8825, 9.0375, 7.23, 5.4225, 9.0375, 9.0375, 3.615, 9.0375, 9.0375, 7.23, 9.0375, 10.845, 5.4225, 10.845, 7.23, 3.615, 12.6525, 7.23, 5.4225, 12.6525, 9.0375, 3.615, 10.845, 7.23, 7.23, 12.6525, 9.0375, 7.23, 10.845, 10.845, 3.615, 12.6525, 10.845, 5.4225, 10.845, 10.845, 7.23]
        cutoff = 2.7
        cellDims = [21.69, 0, 0, 0, 21.69, 0, 0, 0, 21.69]

        connected_points = findConnectivity(pos, cutoff, cellDims)
        expected_connections = [[0, 1], [0, 3], [0, 5], [0, 9], [1, 2], [1, 5], [1, 8], [2, 3], [2, 8], [2, 11], 
         [3, 9], [3, 11], [4, 5], [4, 6], [4, 9], [4, 10], [5, 6], [6, 7], [6, 8], [7, 8], 
         [7, 10], [7, 11], [9, 10], [10, 11], [12, 13], [12, 14], [12, 16], [12, 19], 
         [13, 15], [13, 16], [13, 21], [14, 15], [14, 19], [14, 23], [15, 21], [15, 23], 
         [16, 17], [16, 18], [17, 18], [17, 19], [17, 20], [18, 21], [18, 22], [19, 20], 
         [20, 22], [20, 23], [21, 22], [22, 23]]  # These points should be connected
        self.assertEqual(connected_points, expected_connections)

if __name__ == "__main__":
    unittest.main()