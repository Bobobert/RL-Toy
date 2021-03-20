from RL_Toy.base.const import *

class Q_function:
    """
    Action-value function in hash

    Supports standard method [] for (state, action)
    """

    DEF_ACTION = 0

    def __init__(self):
        self.states = dict()

    def decomposeTuple(self, T):
        state = self.decomposeState(T[0])
        action = self.decomposeAction(T[1])
        return state, action

    @staticmethod
    def decomposeState(S):
        if isinstance(S, dict):
            S = S["agent"]
        return tuple(S)

    @staticmethod
    def decomposeAction(A):
        if not isinstance(A, int):
            A = int(A)
        return A

    def __getitem__(self, state_action):
        state, action = self.decomposeTuple(state_action)
        fromState = self.states.get(state, dict())
        return fromState.get(action, DEF_ACTION)
    
    def __setitem__(self, state_action, value):
        state, action = self.decomposeTuple(state_action)
        fromState = self.states.get(state)
        if fromState is None:
            self.states[state] = {action:value}
        else:
            #action_value = fromState.get(action, 0)
            fromState[action] = value 

    def maxAction(self, state):
        state = self.decomposeState(state)
        actionDict = self.states.get(state,None)
        if actionDict is None:
            return self.DEF_ACTION
        maxV = - np.inf
        maxAction = None
        for action in actionDict.keys():
            action_value = actionDict[action]
            if action_value > maxV:
                maxV = action_value
                maxAction = action
        return maxAction

    def getStates(self):
        return self.states.keys()

def checkForTuple(obj):
    if isinstance(obj, np.ndarray):
        return tuple(obj.tolist())
    elif isinstance(obj, (list, tuple)):
        return tuple(obj)
    else:
        raise TypeError("Object type {} not supported".format(type(obj)))

def toDiscreteSpace(box_space, step: list, limits = None):
    """
    Function to generate a discrete space from a continuos one

    parameters
    ----------
    box_space: gym.spaces.Box
        Expecting a box type of space to generate a discrete one
    step: list
        A list with the step sizes for each dimension. This should match
        the observation_space.shape
    limits: list
        A list with the limits per interval, if None no limits are applied.
        Default is None
    """
    low = box_space.low
    high = box_space.high
    shape = box_space.shape
    spaces = []
    if limits is None:
        limits = [None for i in step]
    for l, h, step_, limit in zip(low, high, step, limits):
        # Calculating how many boxes are needed
        if limit is not None:
            l, h = limit
        boxes = ceil(abs(h - l)/step_)
        spaces += [np.linspace(l, h, num = boxes)]

    return spaces

def cartesian_product(*arrays):
    """
    Cartesian product of ndarrays
    Extracted from https://stackoverflow.com/a/11146645
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)