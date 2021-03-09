from RL_Toy.base.const import *
from RL_Toy.base.basics import Policy
from RL_Toy.utils.functions import toDiscreteSpace, cartesian_product, checkForTuple

class gymPolicy(Policy):
    """
    Base for a gym policy

    parameters
    ----------
    env: gym environment
        The environment
    function: python function
        Must work as function(this, obs), where this is the
        actual policy object and obs the observation. Should return
        the action as the environment acepts it.
    epsilonFunction: python function
        Must work as function(this), where this is the
        actual policy object. Should return a float.
    """
    _eps_ = 0.0
    _eps_test_ = 0.05
    test = False
    greedy = False

    def __init__(self, env, function, epsilonFunction):
        self.env = env
        self.actionSpace = env.action_space
        from gym.spaces import Box, Discrete
        
        if isinstance(self.actionSpace, (Box)):
            self.discrete = False
        elif isinstance(self.actionSpace, (Discrete)):
            self.discrete = True
        
        self.actionFunction = function
        self.epsilonFunction = epsilonFunction

    def _calculate_action(self, obs):
        return self.actionFunction(self, obs)
    
    def getAction(self, obs):
        if (np.random.uniform() < self.epsilon) and not self.greedy:
            return self.actionSpace.sample()
        return self._calculate_action(obs)

    def _get_epsilon(self):
        if self.test:
            return self._eps_test_
        return self.epsilonFunction(self)

    def _set_epsilon(self, newEpsilon):
        if (newEpsilon >= 0) and (newEpsilon <= 1):
            self._eps_ = newEpsilon
        else:
            raise ValueError("New value must be in [0,1]")
        
    epsilon = property(_get_epsilon, _set_epsilon)

class gymPolicyDiscreteFromCon(Policy):
    """
    Gym policy from a continuos observation space and a
    discrete action space

    parameters
    ----------
    env: gym.Environment
        A gym environment type object
    step: list
        A list with the step sizes for each dimension. This should match
        the observation_space.shape
    limits: list
        A list with the limits per interval, if None no limits are applied.
        Default is None
    epsilon: float

    """
    def __init__(self, env, steps:list, limits = None, epsilon:float = 0.0):
        
        self.spaces = toDiscreteSpace(env.observation_space, steps, limits)
        self.observation_space = cartesian_product(*self.spaces)
        self.aS = env.action_space
        self.pi = dict()

        # using the product to make a dictionary
        for i in self.observation_space:
            i = tuple(i.tolist())
            self.pi[i] = self.aS.sample()
        # deleting the product

        self.steps, self.boxes = steps, []
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        if limits is None:
            limits = [None for i in steps]
        else:
            for i, l in enumerate(limits):
                if l is not None:
                    self.low[i] = l[0]
                    self.high[i] = l[1]
        for l, h, step_, limit in zip(self.low, self.high, steps, limits):
        # Calculating how many boxes are needed
            if limit is not None:
                l,h = limit
            self.boxes += [ceil(abs(h - l)/step_)]
        self.epsilon = epsilon
        self.test = False

    def getAction(self, state):
        if (np.random.uniform() > self.epsilon) or self.test:
            tupleState = self.getState(state)
            return self.pi[tupleState]
        return self.aS.sample()

    def update(self, state, action):
        assert self.aS.contains(action), "Action must be contained in the environtment's action space"
        tupleState = self.getState(state)
        self.pi[tupleState] = action 

    def getState(self, state):
        """
        Process a continuos input state into the discrete one. Returns a hashable 
        tuple.
        """
        state = checkForTuple(state)
        assert len(state) == len(self.boxes), "State input must have the same shape as observation_space"
        pos = []
        for s, l, step, b, space in zip(state, self.low, self.steps, self.boxes, self.spaces):
            i = floor((s - l) / step)
            # Bound 
            if i < 0:
                i = 0
            elif i >= b:
                i = b - 1
            pos += [space[i]]
        return tuple(pos)