from .Transitions import *
from .State import *
from . import Minimize

import pickle
from typing import List

class Model:
        
    """
    A class representing a Model.

    Attributes:
        params: The parameters for the model.
        initState: The initial state of the model.
        stateList: A list containing the state objects.
        transitionList: A list containing the transition objects.
        canLabelList: A list of canonical labels.
        SaddleIndices: A list of saddle indices.
        SaddleEnergies: A numpy array of saddle energies.
        Energies: A numpy array of energies.
        Times: A numpy array of times.
        KL: The Kullback-Liebler uncertainty.
        dU_T: A list of dU_T values.

    """

    def __init__(self, params : InputParams):
        
        """
        Initialize the Model class.

        Args:
            params: The parameters for the model.

        """
        
        self.params = params
        self.initState = None

        self.stateList = [] # Contains the state objects
        self.transitionList = [] # Contains the transition Objects
        self.canLabelList = []

    def __len__(self):
        return len(self.transitionList)
   
    def update(self, workDistribution = [], states = [], transitions = [], connections = []):

        ''' want this function to handle both of the above cases for model updating'''

        def cleanData(data):
            return [ x for x in data if x is not None ]
              
        def updateStates(states):
            foundNew = 0
            for s,state in enumerate(states):
                
                if self.checkUniqueness(state):

                    log(__name__,'Added New State to Model')
                    self.stateList.append(state)
                    state.time = self.params.segmentLength
                    foundNew = 1
                else:
                    log(__name__, 'Previously Seen State.')
            
            return foundNew
        
        def updateTransitions(transitions):

            if self.params.maxDefectAtoms == -1:
                maxDefectAtoms = np.inf
            else:
                maxDefectAtoms = self.params.maxDefectAtoms

            foundNew = 0
            for t, transition in enumerate(transitions):
                # HACK: Need to remove the check for maxDefectAtoms....
                if self.checkUniqueness(transition) and transition.initialState.nDefects <= self.params.nDefectsMax and transition.finalState.nDefects <= self.params.nDefectsMax and len(transition.initialState.defectPositions) // 3 <= maxDefectAtoms and len(transition.finalState.defectPositions) // 3 <= maxDefectAtoms:
                    self.transitionList.append(transition)
                    log(__name__,'Added New Transition to Model')
                    foundNew = 1
                    
                    updateStates([transition.initialState, transition.finalState])

                else:
                    # TODO: - this isnt appropriate if the defect seperated
                    log(__name__, 'Previously Seen Transition')

            return foundNew

        # clean the data. it may have NONEs        
        states = cleanData(states)
        transitions = cleanData(transitions)
        connections = cleanData(connections)

        for state in workDistribution: state.time += self.params.segmentLength

        foundNewState = updateStates(states)
        foundNewTrans = updateTransitions(transitions)

        # -1 means no limit so we add everything.
        if self.params.maxModelDepth < 0:
            foundNewConn = updateTransitions([ trans for connection in connections for trans in connection.transitions ])
        
        # TODO: account for different model depth requirements.
        # currently we just add transitions which have an initialState at the starting point.
        else:
            toAdd = []
            for connection in connections:
                for trans in connection.transitions:
                    # TODO: this should probably be done by comparing labels.
                    if maxMoveAtom(trans.initialState, self.initState) < 0.1: 
                        toAdd.append(trans)

            foundNewConn = updateTransitions(transitions = toAdd)

        return max(foundNewState, foundNewConn, foundNewTrans)

    def checkUniqueness(self, obj):
        
        if hasattr(obj, 'NAtoms'):
            targetList = self.stateList
     
        elif hasattr(obj, 'initialState'):
            targetList = self.transitionList

        else:
            sys.exit(TypeError('ERROR: checkUniqueness only accepts State and Transition objects.'))

        if self.params.canonicalLabelling:
            if obj.canLabel in [ target.canLabel for target in targetList ]:
                return 0
            else:
                return 1
        else:
            if obj.nonCanLabel in [ target.nonCanLabel for target in targetList ]:
                return 0
            else:
                return 1
        
    def undecoratedTransitions(self):
        l = []
        for transition in self.transitionList:
            if not transition.redecoration: l.append(transition)
        return l
    
    def workDistribution(self, size):
        inverseTimes = 1 / np.array([ s.time for s in self.stateList ])
        # TODO: account for different model depth requirements.
        if self.params.maxModelDepth > 0: inverseTimes[1:] = 0
        return np.random.choice(self.stateList, p = inverseTimes  / inverseTimes .sum(), size = size)

def checkpoint(model, filename = 'model-checkpoint_latest.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
def setupModel(params, comm = None) -> Model:
    
    """
    Set up the model.

    """
    
    model = Model(params)

    if params.verbose: log(__name__, f"Reading state file: {params.inputFilename}",0)
    model.initState = readStateLAMMPSData(params.inputFilename)

    # mininize State 'in-place'
    Minimize.main(model.initState, params, comm = comm)
    
    model.update(states = [model.initState])

    return model

if __name__ == '__main__':
    pass