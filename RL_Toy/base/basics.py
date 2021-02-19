from RL_Toy.base.const import *

class ActionSpace(ABC):
    """
    Discrete Action Space class.
    Stores all the posible actions for an environment and can generate
    samples with the method sample().

    Parameters
    ----------
    n: int
        Number of total actions. These are consider to be sequential.
    minValue: int
        Default 0. The minimum value that the action can take. It's
        the lower inclusive of the action space intervals 
        [minValue, minValue + n)
    """
    def __init__(self, n:int, minValue:int=0):
        assert n > 0, "Number of actions must be greater than 0"
        self.n = n
        self.mV = minValue
    
    @property
    def shape(self):
        return self.n

    def __iter__(self):
        self.actions = []
        self.i = 0
        for i in range(self.n):
            self.actions += [self.mV + i]
        return self
    
    def __next__(self):
        if self.i == self.n:
            raise StopIteration
        nx = self.actions[self.i]
        self.i += 1
        return nx

    def sample(self):
        """
        Returns a random sample with an uniform distribution of the
        actions.
        """
        return np.random.randint(self.mV, self.mV + self.n)

class ObservationSpace(ABC):
    """
    Discrete observation space class

    Stores all the posible observations for an environment and can generate
    samples with the method sample().

    Parameters
    ----------
    env: Environment
        All the object
    minValue: int
        Default 0. The minimum value that a cell's state can take. It's
        the lower inclusive of the action space intervals 
        [minValue, minValue + n)
    """
    def __init__(self, env:Environment, minValue:int = 0):
        if isinstance(n, int):
            n = (n,)
        assert len(n) > 0, "Tuple must contain at least 1 item"
        self.env = env
        self._n_ = env.shape
        self.mV = minValue

    def __iter__(self):
        self._i_ = []
        self._tl_ = []
        for i in self._n_:
            ax = abs(i - self.mV)
            self._tl_+= [ax - 1] # Max items on this position
            self._i_ += [0] # For actual counter
        self._ii_, self._is_ = 1, 1 # Index position inferior, superior
        return self

    def __next__(self):
        
        def incSupr():
            nonlocal self
            if self._i_[self._is_] == (self._tl_[self._is_]):
                self._i_[self._is_] = 0 # reset superior index
                self._is_ += 1 # increment one
                if self._is_ == len(self._n_):
                    raise StopIteration
                incSupr()
            else:
                self._i_[self._is_] += 1
                self._is_ = 1

        def zeroCnt():
            nonlocal self
            if self._i_[0] == (self._tl_[0]):
                self._i_[0] = 0 # Reset the low index
                incSupr()
            else:
                self._i_[0] += 1

        def doTpl():
            nonlocal self
            tupl = []
            for i in self._i_:
                tupl += [self.mV + i]
            return tuple(tupl)

        if len(self._n_) < 2:
                if self._i_[0] <= (self._tl_[0]):
                    t = doTpl()
                    self._i_[0] += 1
                    return t
                else:
                    raise StopIteration
        else:
            zeroCnt()
        tpl = doTpl()
        if not self.env.isValid(tpl):
            return __next__()
            
        return tpl


    @property
    def shape(self):
        return self._n_

    def sample(self):
        """
        Returns a random sample with an uniform distribution of the
        cell's states.
        """
        newState = []
        for i in self._n_:
            newState += [np.random.randint(self.mV, self.mV + i)]
        return newState


class Environment(ABC):
    """
    Environment base class.
    """
    def step(self, action):
        """
        Executes the action, updates the environment, calculates 
        the reward and observation output.
        Returns
        -------
        observation, reward, done
        """
        raise NotImplementedError
    def reset(self):
        """
        Restart the initial state of the environment, 
        in a deterministic or stochastic manner
        Returns
        ------
        obeservation
        """
        raise NotImplementedError
    def getObservation(self):
        """
        Calculates and returns the observation of 
        the actual state of the environment.
        """
        raise NotImplementedError
    def calculateReward(self, state):
        """
        Calculate with the actual mode the reward 
        from the last observation made in the environment

        Returns
        -------
        reward
        """
        raise NotImplementedError
    @property
    def actionSpace(self):
        """
        Returns the ActionShape class designed of the environment.
        """
        raise NotImplementedError
    @property
    def observationSpace(self):
        """
        Returns a list or generator of all the states availables. 
        """
        raise NotImplementedError
    def transProb(self, state, action):
        """
        Returns the probabilities and states of the transitions from RL_Toy.the
        state and action given.
        """
        raise NotImplementedError
    def isTerminal(self, state):
        """
        Returns the bool that expreses if 
        the actual state is a terminal one
        or not.
        """
        raise NotImplementedError

class Policy(ABC):
    """
    Policy base class.
    """
    def getAction(self, state):
        """
        Calculates and returns the corresponding 
        action for the state given.
        """
        raise NotImplementedError
    def update(self, state, action):
        """
        Update the action per state manner of the policy
        """
        raise NotImplementedError