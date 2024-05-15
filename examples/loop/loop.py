# P
from HopDec.ASE import *
from HopDec.Input import *
from HopDec.Lammps import *
from HopDec.State import *
from HopDec.Model import *
from HopDec.Utilities import log, writeTerminalBlock
import HopDec.Redecorate as Redecorate
import HopDec.Minimize as Minimize
import HopDec.NEB as NEB
import HopDec.MD as MD
import HopDec.Dimer as DIMER

#
import copy
import pickle
import random
import pandas as pd

def main_genNewConnection(params : InputParams, model = None, baseComm = None, method = 'MD'):

    '''
    ################################################################################################################################################
    ################## PETE INITIAL CONNECTION #####################################################################################################
    ################################################################################################################################################
    '''

    connection = None
    state = None
    comm = None
    color = 0
    flagFound = 0
    flagArray = []

    if baseComm:
        color = rank
        comm = baseComm.Split(color = color, key = rank)
        
    # setup a model if we aren't given one.
    if not model:   
        model = Model(params)
        model.setup(comm = comm)
    
    # run segments until somebody finds a hop we havent seen before.
    while True:

        baseComm.barrier()

        chosenState = random.choice(model.stateList)
        state = copy.deepcopy(chosenState)

        if method == 'MD':

            # RUN MD BLOCK
            # find new state with MD - MD.main runs MD in segments of 2 ps at 1200 K
            log('MD', f'rank {rank}: Running MD in state: {state.canLabel}',1)   
            _, state, flag = MD.main(state, params, comm = comm)
            
            # update model times after we did our MD
            chosenState.time += params.segmentLength

        # flag if we did any hop
        if flag:
            
            # get hash of the transition (union of the initial and final)
            getStateCanonicalLabel(state, params, comm = comm)
            currentTransHash = getTransitionHash(params, chosenState, state)
            
            # test if its a new transition
            if currentTransHash not in [ trans.transitionHash for trans in model.transitionList ]:

                log('NEB', f'rank: {rank} Running NEB', 1)  

                connection = NEB.main(chosenState, state, params, exportStructures = False, plotPathways = False, comm = comm, verbose = False)

                for t,transition in enumerate(connection.transitions):

                    # get hash of the transition (union of the initial and final)
                    currentTransHash = getTransitionHash(params, transition.initialState, transition.finalState)
                    transition.hash = currentTransHash

                    # check if the NEB slicing discovered something new
                    if currentTransHash not in [ trans.transitionHash for trans in model.transitionList ]:
                        
                        log('Model', 'New Transition')

                        model.addState(state,comm = comm)
                        model.addTransitionTrans(transition)
                        flagFound = 1
                    else:
                        log('Model', 'Previously Seen Connection')
            else:
                    log('Model', 'Previously Seen Connection')

        baseComm.barrier()

        # gather and broadcast the flags from everybody
        flagArray = baseComm.gather(flagFound, root = 0)
        flagArray = baseComm.bcast(flagArray, root = 0)

        # tell everybody to stop if we find something and propogate new model to everybody
        if 1 in flagArray:
            finder = np.where(np.array(flagArray) == 1)[0][0]

            # NEED TO VERIFY THIS:
            # model = baseComm.bcast(model, root = finder)
            # IS EQUIVALENT TO THIS:
            model.transitionList = baseComm.bcast(model.transitionList, root = finder)
            model.stateList = baseComm.bcast(model.stateList, root = finder)
            model.canLabelList =  baseComm.bcast(model.canLabelList, root = finder)
            break

    comm.Free()

    return model

def main_RedecorateTransition(params : InputParams, transition : Transition, baseComm = None):

    # Given that we have found a valid Connection / Transition. We redecorate 'nDecorations' many times
#    params.nDecorations = 10
    redecResults = Redecorate.main(transition.initialState, transition.finalState, params, pickle = True, comm = baseComm)

    # flag to indicate we have decorated this transition
    transition.redecorated = 1

    return str(redecResults.transHash) + '.pickle'

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################


if __name__== "__main__":

    from mpi4py import MPI

    model = None
    refStructure = 'data/CuSupercell_CleanReference.dat'

    ### Comm Setup ###
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()    

    if not rank: writeTerminalBlock('DR Closed Loop Prototype')

    # read HopDec parameters
    params = getParams()

    while True:

        # find a transition (connection) to somewhere new...
        model = main_genNewConnection(params, model = model, baseComm = comm, method = 'MD')

        # redecorate the undecorated Transitions
        undecoratedTransitions = model.getUndecTrans()
        dataPkls = []
        for trans in undecoratedTransitions:
            pklFilename = main_RedecorateTransition(params, trans, baseComm = comm)
            dataPkls.append(pklFilename)

    # serial
    # if not rank:
        
        # train the ML model with generated MD data
        # main_ML(pklFilename, refStructure)
    
        # Matts KMC
        # main_KMC()

        # refine some barriers which Matt says we are uncertain about.
        # pklFilename = refineBarriers('event_configs.pkl', params)

        # update the ML model with generated NEB data
        # main_ML('dummyNEBCompleted.pickle', refStructure, run_type='update_model')

    comm.barrier()
    if not rank: writeTerminalBlock('Fin.')

    # clean up
    MPI.Finalize()


