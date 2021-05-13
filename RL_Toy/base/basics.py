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

    def __len__(self):
        return self.n

    def sample(self):
        """
        Returns a random sample with an uniform distribution of the
        actions.
        """
        return np.random.randint(self.mV, self.mV + self.n)


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

    test = False
    greedy = False
    _eps_ = 0.0
    
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
    
    @property
    def epsilon(self):
        return self._eps_
    
    @epsilon.setter
    def epsilon(self, X):
        if (X >= 0) and (X <= 1):
            self._eps_ = X
        else:
            print("Invalid Epsilon, remain the same {}".format(self._eps_))


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
            return self.__next__()
            
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
        
class Agent:
    """
    Base object for an agent. Which works as container
    for the policy, the environment and a middle ground
    for any particular processing between those for the 
    RL algorithm.
    
    Method test is not recomended to be modified.
    """
    policy = None
    env, env_test = None, None
    name = "base_Agent_v0"
    done, lastObservation = True, None
    episodeSteps, episodeReward = 0, 0.0
    def __init__(self):
        assert self.env is not None, \
            "Agent needs environment reference"
        assert self.policy is not None, \
            "Agent needs policy object"

    def processObs(self, obs):
        """
        Method for each agent to process the
        observation if need for the policy or another
        further methods
        """
        return obs
    
    def processAction(self, action_raw):
        """
        If needed, this method should process and adecuate
        the action signla properly for the environment
        """
        return action_raw

    def processReward(self, reward_raw):
        """
        If need, this method should process the 
        reward, still returns (int or float)
        """
        return reward_raw

    def step(self, **kwargs):
        """
        Execute a step on the environment with 
        the given policy
        
        returns
        action, reward, steps, done, info
        """
        env, pi = self.env, self.policy
        if self.done:
            obs = env.reset()
            self.done = False
            self.episodeReward = 0.0
            self.episodeSteps = 0
        else:
            obs = self.lastObservation

        state = self.processObs(obs)
        action_raw = pi.getAction(state)
        action = self.processAction(action_raw)
        nextObs, reward, done, info = env.step(action)

        # Save reward
        r = self.processReward(reward)
        self.episodeSteps += 1
        self.episodeReward += r

        self.lastObservation = nextObs
        self.done = done

        return state, action, reward, self.episodeSteps, done, info
    
    def test(self, **kwargs):
        """
        Execute a test on the environment with 
        the actual policy
        """
        self.testMode(True)
        pi = self.policy

        if self.env_test is None:
            env = self.env
            self.done = True
        else:
            env = self.env_test

        n_test = kwargs.get("n_test", 10)
        tests_results, tests_steps = [], []
        for i in range(n_test):
            done = False
            obs = env.reset()
            test_return, test_steps = 0.0, 0
            while not done:
                state = self.processObs(obs)
                ar = pi.getAction(state)
                obs, reward, done, _ = env.step(self.processAction(ar))
                test_return += reward
                test_steps += 1
            tests_results += [test_return]
            tests_steps += [test_steps]

        self.testMode(False)

        return tests_results, tests_steps

    def update(self, obs, action):
        """
        This method should be modified if one is expecting
        to process the information before sending it to the
        policy. Otherwise should work properly
        """
        state = self.processObs(obs)
        action = self.processAction(action)
        self.pi.update(state, action)
        
    def getAction(self, obs):
        """
        This method returns the action from the policy
        """
        state = self.processObs(obs)
        return self.policy.getAction(state)
        
    def testMode(self, mode : bool = True):
        """
        If the policy supports it, its changed to 
        test mode if mode is True.
        """
        try:
            self.policy.test = mode
        except AttributeError:
            pass
        
        
class AgentToy(Agent):
    """
    Based on Agent with exceptions for the lack of
    info reeturn from the RL_Toy environments.
    """
    def step(self, **kwargs):
        """
        Execute a step on the environment with 
        the given policy
        
        returns
        action, reward, steps, done, info
        """
        env, pi = self.env, self.policy
        if self.done:
            obs = env.reset()
            self.done = False
            self.episodeReward = 0.0
            self.episodeSteps = 0
        else:
            obs = self.lastObservation

        state = self.processObs(obs)
        action_raw = pi.getAction(state)
        action = self.processAction(action_raw)
        nextObs, reward, done = env.step(action)

        # Save reward
        r = self.processReward(reward)
        self.episodeSteps += 1
        self.episodeReward += r

        self.lastObservation = nextObs
        self.done = done

        return state, action, reward, self.episodeSteps, done 
    
    def test(self, **kwargs):
        """
        Execute a test on the environment with 
        the actual policy
        """
        self.testMode(True)
        pi = self.policy

        if self.env_test is None:
            env = self.env
            self.done = True
        else:
            env = self.env_test

        n_test = kwargs.get("n_test", 10)
        tests_results, tests_steps = [], []
        for i in range(n_test):
            done = False
            obs = env.reset()
            test_return, test_steps = 0.0, 0
            while not done:
                state = self.processObs(obs)
                ar = pi.getAction(state)
                obs, reward, done = env.step(self.processAction(ar))
                test_return += reward
                test_steps += 1
            tests_results += [test_return]
            tests_steps += [test_steps]

        self.testMode(False)

        return tests_results, tests_steps
    