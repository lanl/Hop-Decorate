import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from functools import partial
import networkx as nx
import copy
from typing import List

from .State import *
import HopDec.Minimize as Minimize
from .Transitions import *
from . import Constants

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

        self.SaddleIndices = []
        self.SaddleEnergies =  np.array([])
        self.Energies = np.array([])
        self.Times = np.array([])

        self.KL = np.inf
        self.dU_T = []

        self.steps = 0
        self.clock = 0 # ps

    def __len__(self):
        return len(self.transitionList)

    def setup(self, comm = None) -> None:
        
        """
        Set up the model.

        """
        if self.params.verbose: log(__name__, f"Reading state file: {self.params.inputFilename}",1)
        self.initState = readStateLAMMPSData(self.params.inputFilename)

        # mininize State 'in-place'
        lmp = LammpsInterface(self.params, communicator = comm)
        Minimize.main(self.initState, lmp, self.params)
        
        self.addState(self.initState,comm=comm)

    def addTransitionTrans(self, transition):
        _ = self.addTransition(transition.initialState, transition.finalState, transition.forwardBarrier, transition.hash)

    def addTransition(self, initialState : State, finalState : State, barrier : float, hash : str) -> Transition:
        
        """
        Add a transition to the model.

        Args:
            initialState: The initial state for the transition.
            finalState: The final state for the transition.
            barrier: The barrier energy for the transition.
            hash: The hash value for the transition.

        Returns:
            Transition: The transition object added to the model.

        """

        canLabels = np.array([ state.canLabel for state in self.stateList ])
        self.SaddleIndices.append([np.min([np.where(canLabels == initialState.canLabel)[0]]), np.max([np.where(canLabels == finalState.canLabel)[0]] ) ])
        
        trans = Transition(initialState, finalState)
        trans.forwardBarrier = barrier

        trans.initialHash = initialState.canLabel
        trans.finalHash = finalState.canLabel

        trans.transitionHash = hash
        self.transitionList.append(trans)

        return trans

    def addState(self, state,comm = None) -> State:

        """
        Add a state to the model.

        Args:
            pos: The positions for the new state.

        Returns:
            State: The state object added to the model.

        """

        getStateCanonicalLabel(state, self.params, comm=comm)

        if state.canLabel in self.canLabelList:# and len(self.canLabelList) > 1:
            if self.params.verbose: log(__name__, 'Symmetric State Discovered.')
            return 0
        else:
            if self.params.verbose: log(__name__,'Added New State to Model')
            self.stateList.append(state)
            self.canLabelList.append(state.canLabel)
            state.time = 2000
            return 1

        # return state

    def getUndecTrans(self):
        l = []
        for transition in self.transitionList:
            if not transition.redecorated: l.append(transition)
        return l

    def plot(self, name = 'network'):
        
        """
        Plot the network with networkx.

        Args:
            name: The filename for the plotted network.

        """
                
        fig,axs = plt.subplots(figsize=(5, 5))

        G = nx.from_edgelist( [ (connection[0].item(), connection[1].item()) for connection in self.SaddleIndices ] )

        for i in range(len(self.SaddleIndices)):
            for j in range(2):
                axs.text(self.SaddleIndices[i][j], self.SaddleIndices[i][j], self.canLabels[j], ha='center', va='bottom', fontsize = 6)

        axs.scatter( G.nodes , G.nodes, s = 100)
        for i in range(len(self.SaddleIndices)):
            axs.plot(self.SaddleIndices[i], self.SaddleIndices[i], linewidth = 3, alpha = 0.5)

        axs.margins(x=20, y=20)
        axs.set_xlim([-0.5,2]) 
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(f"{name}.pdf")

    # def pullData(self):
        
    #     self.Energies = jnp.array([s.totalEnergy for s in self.stateList])
    #     Energies = self.Energies - jnp.min(self.Energies)

    #     Times = jnp.array([s.time for s in self.stateList], dtype=float)

    #     saddleEnergies = jnp.array([ t.forwardBarrier for t in self.transitionList ])

    #     return Energies, Times, saddleEnergies

    # def resourceAllocation(self):
    
    #     """
    #     Calculate the optimal resource allocation using KL divergence.

    #     """

    #     # TODO: I still think there are things going wrong here...

    #     SaddleIndices = self.SaddleIndices
    #     Energies, Times, SaddleEnergies = self.pullData()
    #     Temperature = self.params.MDTemperature

    #     def skeleton_matrix(energies, saddle_indices):
    #         N = len(energies)
    #         G = np.zeros((N, N, len(saddle_indices)))
    #         S = np.zeros((N, N, len(saddle_indices)))
    #         for i, (j, k) in enumerate(saddle_indices):
    #             G[j][k][i] = energies[k]
    #             G[k][j][i] = energies[j]
    #             S[j][k][i] = 1.0
    #             S[k][j][i] = 1.0
    #         return G, S

    #     def make_unseen_rate(k, t, N):
    #         A = jnp.zeros(N)
    #         build_A = jit(jax.vmap(lambda kl: (k * kl / (k + kl)).sum()))
    #         A = build_A(k)
    #         A = jnp.where(jnp.isnan(A), 0.0, A)
    #         C = jnp.ones(N + 1)
    #         for a in A:
    #             C = jnp.roll(C, -1) + a * C

    #         # list of factorials
    #         inc_fac = 1.0 * jnp.append(jnp.ones(1), jnp.arange(1, N + 1).cumprod())
    #         tpow = jnp.exp(-jnp.arange(N + 1) * jnp.log(t))
    #         norm = (inc_fac * jnp.flip(C) * tpow / t).sum()

    #         inc_fac2 = 1.0 * jnp.arange(1, N + 2).cumprod()
    #         tpow = jnp.exp(-jnp.arange(1, N + 2) * jnp.log(t))
    #         ku = (inc_fac2 * jnp.flip(C) * tpow / t).sum() / norm
    #         return ku

    #     def rate_matrix(SE,Times):
    #         K = Ko(SE)
    #         return K - jnp.diag(K.sum(0) + ku(K, Times))

    #     def MLE_K(SE):
    #         K = Ko(SE)
    #         K -= np.diag(K.sum(0))
    #         nu, v, w = eig(K, left=True)
    #         # normalization factor
    #         dp = np.sqrt(np.diagonal(w.T.dot(v))).real
    #         # dot product (v.P(0)=rho)
    #         v = (v.real.dot(np.diag(1.0 / dp))).T
    #         # dot product (1.T.w)
    #         w = w.real.dot(np.diag(1.0 / dp))
    #         nu = nu.real
    #         return w, nu, v

    #     def KullbackLieblerUncertainty(SE,Times):
    #         K = rate_matrix(SE,Times)
    #         P0 = np.ones(nu.size)
    #         t = -jnp.linalg.solve(K, P0).sum()
    #         Pbar = w @ jnp.diag(jnp.exp(nu * t)) @ (v @ P0)
    #         nu_true = jnp.diag(v @ K @ w)
    #         P = w @ jnp.diag(jnp.exp(nu_true * t)) @ (v @ P0)
    #         return -jnp.log(P / Pbar) @ Pbar

    #     dKullbackLieblerUncertainty = grad(KullbackLieblerUncertainty,argnums=1,allow_int=True)

    #     G, S = skeleton_matrix(Energies, SaddleIndices)

    #     Ko = jit(partial(lambda SE, T, G, S: jnp.exp(-(jnp.einsum('ijk,k->ijk', S, SE) - G).sum(-1) / Constants.boltzmann / T).T, G=G, S=S, T=Temperature))
    #     w, nu, v = MLE_K(SaddleEnergies)
    #     ku = jax.vmap(jax.jit(partial(make_unseen_rate,N=len(Energies))),in_axes=(0,0))

    #     dKullbackLieblerUncertainty_bar = jax.grad(KullbackLieblerUncertainty,argnums=0)
    #     dKullbackLieblerUncertainty_T = jax.grad(KullbackLieblerUncertainty,argnums=1,allow_int=True)

    #     U = KullbackLieblerUncertainty(SaddleEnergies,Times)
    #     self.KL = U

    #     dU_bar = dKullbackLieblerUncertainty_bar(SaddleEnergies,Times)
    #     self.dU_T = dKullbackLieblerUncertainty_T(SaddleEnergies,Times)
    #     return self.dU_T
    
    def roulette(self, temperature):
        
        # collect rates of found transitions
        totalRate = 0
        rates = []
        for trans in self.transitionList:
            r = trans.calcRate(temperature)
            rates.append(r)
            totalRate += r

        spin = random.choices(range(len(self.transitionList)), weights = rates)[0]

        return spin, self.transitionList[spin].finalState, 0
    
    def takeStep(self, state, t):
        self.stateList.append(state)
        self.clock += t
        
def getTransitionHash(params : InputParams, initialState : State, finalState : State) -> str:

    """
    Get the hash value for a transition.

    Args:
        params: The input parameters.
        initialState: The initial state for the transition.
        finalState: The final state for the transition.

    Returns:
        str: The hash value for the transition.

    """

    dummyState = copy.deepcopy(initialState)
    dummyState.NAtoms = int((len(initialState.defectPositions)+len(finalState.defectPositions))//3)
    dummyState.defectPositions = np.concatenate((initialState.defectPositions, finalState.defectPositions))
    dummyState.pos = np.concatenate((initialState.defectPositions, finalState.defectPositions))
    defectTypes = np.concatenate((initialState.defectTypes, finalState.defectTypes))
    
    graphEdges = findConnectivity(dummyState.defectPositions, params.bondCutoff, dummyState.cellDims)
    
    hash = canLabelFromGraph(graphEdges, defectTypes)
    
    dummyState = None
    return hash
    
if __name__ == '__main__':
    pass