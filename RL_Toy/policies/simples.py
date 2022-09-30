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
    """
    Arguments:
    env:
        Expects an environment RL-Toy or gym type would be nice.
    epsilon:
        Any float in the range [0, 1].
    epsilon_min:
        Any float in the range [0, 1].
    mode:
        Optional. A string from options: linear, exponential, none. For the epsilon decay
    mode_steps:
        Integer for the mode of the epsilon decay.
    """
    def __init__(self, env: Environment, 
                    epsilon: float = 0.1,
                    epsilon_min: float = 0.0,
                    mode: str = 'none', 
                    mode_steps: int = 10 ** 3):
        assert (epsilon >= 0) and (epsilon <= 1), "Epsilon needs to be a float in [0,1]"
        super(gridPolicyEpsilon, self).__init__(env)
        self.epsilon = self.epsilon_init = epsilon
        self.epsilon_min = epsilon_min if 1 >= epsilon_min >= 0.0 else 0.0
        self.epsilon_mode = mode if mode in ['linear', 'exponential'] else 'none'
        self.mode_steps = mode_steps if mode_steps >= 0 else 0

        self._action_calls = 0
        self.test = False
        self.epsilon_prev = 0.0

    def getAction(self, state):
        if self.test:
            self._epsilon_test_()
        else:
            self._epsilon_decay_()

        throw = np.random.uniform()
        if throw < self.epsilon:
            action = self.env.actionSpace.sample()
        else:
            if isinstance(state, dict):
                state = state["agent"]
            action = self.pi[state]
        return action

    def _epsilon_decay_(self):
        if self.epsilon_mode == 'none':
            new_epsilon = self.epsilon
        elif self.epsilon_mode == 'linear':
            new_epsilon = (self.mode_steps -  self._action_calls) / self.mode_steps * self.epsilon
        elif self.epsilon_mode == 'exponential':
            new_epsilon = exp(-self._action_calls / self.mode_steps) * self.epsilon
        else:
            raise AttributeError('How? Epsilon mode is not valid')

        self.epsilon = max(self.epsilon_min, new_epsilon)
        self._action_calls += 1

    def _epsilon_test_(self):
        # Silly way to save it, assuming that epsilon test is always the min!
        self.epsilon_prev = max(self.epsilon, self.epsilon_prev)
        self.epsilon = self.epsilon_min