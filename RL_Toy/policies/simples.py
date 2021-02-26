from RL_Toy.base import Policy, Environment
from RL_Toy.base.const import *

class uniformRandomPolicy(Policy):
    def __init__(self, env:Environment):
        self.pi = env.actionSpace
        self.env = env

    def getAction(self, state):
        return self.pi.sample()

    def update(self, state, action):
        pass # Do nothing

class gridPolicy(Policy):
    def __init__(self, env:Environment):
        self.pi = np.zeros(env.shape, dtype=UINT_DEFT)
        # or could be a dict() as well
        self.env = env
        self.randomInit()

    def randomInit(self):
        for state in self.env.observationSpace:
            self.pi[state] = self.env.actionSpace.sample()

    def update(self, state, action):
        if isinstance(state, dict):
            state = state["agent"]
        self.pi[state] = action

    def getAction(self, state):
        if isinstance(state, dict):
            state = state["agent"]
        return self.pi[state]

class gridPolicyEpsilon(gridPolicy):
    def __init__(self, env: Environment, epsilon: float = 0.1):
        assert (epsilon >= 0) and (epsilon <= 1), "Epsilon needs to be a float in [0,1]"
        super(gridPolicyEpsilon, self).__init__(env)
        self.epsilon = epsilon

    def getAction(self, state):
        throw = np.random.uniform()
        if throw < self.epsilon:
            action = self.env.actionSpace.sample()
        else:
            if isinstance(state, dict):
                state = state["agent"]
            action = self.pi[state]
        return action