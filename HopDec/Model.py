from .Transitions import *
from .State import *
from . import Minimize

import pickle as pkl
from typing import List
import networkx as nx

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

        self.graph = None

        self.stateList = [] # Contains the state objects
        self.transitionList = [] # Contains the transition Objects
        self.canLabelList = []
        self.stateWorkCheck = np.array([]) # 0 = do work, 1 = dont do work
        

    def __len__(self):
        return len(self.transitionList)
    
    def loadRedecorations(self):
        # TODO: Add logging
        self.redecorations = []
        for trans in self.transitionList:
            df = trans.loadRedecoration()

            if df.empty:
                self.redecorations.append(None)
            else:
                self.redecorations.append(copy.copy(df))
    
    def buildModelGraph(self):

        edges = [ ( trans.initialState.nonCanLabel, trans.finalState.nonCanLabel ) 
                    for trans in self.transitionList ]
        
        nodes = [ state.nonCanLabel for state in self.stateList ]

        self.graph = buildNetwork(nodes, edges)

    def findDepth(self, state):        
        return shortestPath(self.graph, self.initState.nonCanLabel, state.nonCanLabel)
        
    def update(self, workDistribution = [], states = [], transitions = [], connections = []):

        def cleanData(data):
            return [ x for x in data if x is not None ]
              
        def updateStates(states):
            foundNew = 0
            for s,state in enumerate(states):
                
                if self.checkUniqueness(state):

                    self.stateList.append(state)
                    self.buildModelGraph()
                    depth = self.findDepth(state)

                    # This is a catch for if we find a state but cant resolve a transition to it.
                    if depth == np.inf:
                      print('WARNING: Found State with no valid Transition. Skipping...')
                      _ = self.stateList.pop(-1)
                      continue

                    if depth <= self.params.maxModelDepth or self.params.maxModelDepth < 0: 
                        state.doWork = 1
                    else:
                        state.doWork = 0

                    state.time = self.params.segmentLength
                    log(__name__,'Added New State to Model')
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
                    # TODO: - this message isnt appropriate if the defect seperated
                    log(__name__, 'Previously Seen Transition')

            return foundNew
        
        # clean the data. it may have NONEs        
        states = cleanData(states)
        transitions = cleanData(transitions)
        connections = cleanData(connections)

        # update MD time in each state where MD was done.
        for state in workDistribution: state.time += self.params.segmentLength

        # generally not used during 'HopDec-main' functionality
        foundNewTrans = updateTransitions(transitions)
        foundNewState = updateStates(states)

        # When updating the model during 'HopDec-main' 
        # we are usually given a Connection object which is handled below.

        # -1 means no limit so we attempt to add every transition to the model.
        if self.params.maxModelDepth < 0:
            foundNewConn = updateTransitions([ trans for connection in connections for trans in connection.transitions ])
        
        # otherwise we need to check state depths.
        else:

            toAdd = []
            foundNewConn = 0

            for connection in connections:

                self.buildModelGraph()

                for t,trans in enumerate(connection.transitions):

                    if self.findDepth(trans.initialState) <= self.params.maxModelDepth:

                    # if currentDepth <= self.params.maxModelDepth: # if the initial state is one in which we would like to search                

                        # toAdd.append(trans)
                        test = updateTransitions(transitions = [trans])
                        foundNewConn = max(foundNewConn,test)

            # foundNewConn = updateTransitions(transitions = toAdd)

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
    
    def workDistribution(self, size):

        inverseTimes = 1 / np.array([ s.time for s in self.stateList ])
        workArray = np.array([ s.doWork for s in self.stateList ])
        inverseTimes = inverseTimes * workArray

        return np.random.choice(self.stateList, p = inverseTimes  / inverseTimes .sum(), size = size)

def checkpoint(model, filename = 'model-checkpoint_latest.pkl'):
    with open(filename, 'wb') as f:
        pkl.dump(model, f, protocol=4)
    
def setupModel(params, comm = None) -> Model:
    
    """
    Set up the model.

    """
    
    model = Model(params)

    if params.verbose: log(__name__, f"Reading state file: {params.inputFilename}",0)
    model.initState = readStateLAMMPSData(params.inputFilename)

    Minimize.main(model.initState, params, comm = comm)
    
    model.update(states = [model.initState])

    return model

if __name__ == '__main__':
    pass