from RL_Toy.base.const import *
from RL_Toy.base.basics import Policy

class gymPolicy(Policy):
"""
Base for a gym policy
"""
    _eps_ = 0.0
    _eps_test_ = 0.05
    # This is meant to hold a epsilon modifier
    epsilonFunction = lambda x: x._eps_

    def __init__(self, env):
        self.env = env
        self.actionSpace = env.action_space
        from gym.spaces import Box, Discrete
        
        if isinstance(self.actionSpace, (Box)):
            self.discrete = False
        elif isinstance(self.actionSpace, (Discrete)):
            self.discrete = True

    def _calculate_action(self, obs):
        raise NotImplementedError
    
    def getAction(self, obs):
        if (np.random.uniform() < self.epsilon) and not self.greedy:
            return self.actionSpace.sample()
        return self._calculate_action(obs)

    @property
    def epsilon(self):
        if self.test:
            return self._eps_test_
        return self.epsilonFunction(self)

    @epsilon.setter
    def _set_episilon(self, newEpsilon):
        assert (newEpsilon >= 0) and (newEpsilon <= 1), "New value must be in [0,1]"
        self._eps_ = newEpsilon