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
        self.pi = np.zeros(env.shape, dtype=np.uint8)
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