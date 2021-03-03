class Q_function():
    """
    Action-value function in hashables.
    """

    DEF_ACTION = 5

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
        return fromState.get(action, 0)
    
    def __setitem__(self, state_action, value):
        state, action = self.decomposeTuple(state_action)
        fromState = self.states.get(state)
        if fromState is None:
            self.states[state] = {action:value}
        else:
            action_value = fromState.get(action, 0)
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