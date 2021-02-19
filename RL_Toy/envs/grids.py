from RL_Toy.base import  Environment, ActionSpace, ObservationSpace
from RL_Toy.base.const import *

class gridWorld(Environment):
    """
    Little and simple environment for a deterministic grid world.
    Parameters
    ----------
    width: int
        First dimension of the grid
    height: int
        Second dimension of the grid
    initPos: tuple of int
        Initial position of the agent.
    goal: tuple of int
        Position of the first goal to create the grid.
        One can add more goals later if required.
    movement: str
        Default 8C. Refer to step method.
    horizon: int
        Default 10**6. Number of steps to run the environment before it
        terminates.
    """
    # All gfx related
    EMPTYC = (255, 255, 255)
    OBST = 2
    OBSTC = (128, 64, 64)
    VORTEX = 3
    VORTEXC = (230, 0, 10)
    GOAL = 1
    GOALC = (0, 200, 20)
    AGENTC = (230, 150, 240)
    POLICYC = (0, 1, 0.5)
    CELLSIZE = 4
    GRAPHSCALE = 1.2    

    VORTEXD = [[False, True, True, False],
             [True, False, False, True],
             [True, False, False, True],
             [False, True, True, False]]

    AGENTD = [[False, True, True, False],
              [True, True, True, True],
              [True, True, True, True],
              [False, True, True, False]]

    GOALD = [[True, False, True, False],
             [False, True, False, True],
             [True, False, True, False],
             [False, True, False, True]]

    # Action meaning related
    actions  = [(-1,-1),(-1, 0),(-1,1),
              (0, -1),(0, 0),(0, 1),
              (1, -1),(1, 0),(1, 1)]

    actions4C = [1,3,4,5,7]

    def __init__(self, width:int, height:int, initPos:tuple, goal:tuple, movement:str = "4C", horizon:int = 10**6):
        # Grid Related
        self.grid = np.zeros((width, height), dtype=np.uint8)
        self._w = width
        self._h = height
        self.obstacles = []
        self.vortex = []
        self.goal = [goal]
        self.steps = 0
        self.gameOver = False
        self.horizon = horizon
        # Agent related
        self.movMode = movement
        self.validateTuple(initPos)
        self.initX, self.initY = initPos
        self.posX, self.posY = initPos
        self.__actionSpace = ActionSpace(9 if movement == "8C" else 5, 1)
        self._obsSpace = ObservationSpace(self.grid.shape)
        # Graphics related
        self.frame = np.zeros((width * self.CELLSIZE, height * self.CELLSIZE, 3), dtype=np.uint8)
        # Initialize the grid
        self.reset()

    def validateTuple(self, T:tuple):
        assert len(T) == 2, "Tuple needs to have 2 items for x and y"
        if (T[0] < 0) or (T[1] < 0):
            raise ValueError("Values of the tuple must be non-negative")
        if (T[0] >= self._w) or (T[1] >= self._h):
            raise ValueError("Value of the tuple need to be in the interval x[0, {}), y[0, {})".format(self._w, self._h))
        return True
    
    def addVortex(self, *vortex):
        """
        Add a vortex on the grid
        Parameters
        ---------
        vortex: tuple of int
            A tuple of integers that cointains the position in which one 
            desire to put a new vortex.
        """
        for v in vortex:
            self.validateTuple(v)
            self.vortex += [v]

    def addObstacles(self, *obstacles):
        """
        Add an obstacle on the grid
        Parameters
        ---------
        obstacles: tuple of int
            A tuple of integers that cointains the position in which one 
            desire to put a new obstacle.
        """
        for o in obstacles:
            self.validateTuple(o)
            self.obstacles += [o]
    
    def addGoals(self, *goals):
        """
        Add a goal on the grid
        Parameters
        ---------
        goal: tuple of int
            A tuple of integers that cointains the position in which one 
            desire to put an additional goal.
        """
        for g in goals:
            self.validateTuple(g)
            self.goal += [g]

    def reset(self, initialPos = None):
        self.grid[:,:] = 0
        for o in self.obstacles:
            self.grid[o] = self.OBST
        for v in self.vortex:
            self.grid[v] = self.VORTEX
        for g in self.goal:
            self.grid[g] = self.GOAL
        if initialPos is None:
            self.posX = self.initX
            self.posY = self.initY
        else:
            self.validateTuple(initialPos)
            self.posX, self.posY = initialPos
        self.steps = 0
        self.gameOver = False
        self.lastObs = self.getObservation()
        self.lastReward = 0
        self.lastAction = 5
        return self.lastObs

    def step(self, action:int = 5):
        """
        Excecute a step on the environment. 
        The actions on the grid that the agent can take on mode 8C are
        integers from 1 to 9.

            [1  2  3]
            |4  5  6|
            [7  8  9]

        5 being the neutral action or "Do nothing"

        In mode 4C, the action space is reduced to just move in a cross 
        pattern with integers from 1 to 5

            [-  1  -]
            |2  3  4|
            [-  5  -]
        
        3 being the "do nothing" action.

        Parameters
        ----------
        action: int
            
        Returns
        -------
        observation , reward, done
        """
        # If the environment has reached a terminal state
        if self.gameOver:
            return self.lastObs, 0, True
        # Select the action from the corresponding transition probabilities
        randomSelect = np.random.uniform(0,1)
        probs, states = self.transProb(self.lastObs, action)
        lastP = 0
        for p, s in zip(probs, states):
            if randomSelect <= (p + lastP):
                self.posX, self.posY = s
                break
            else:
                lastP += p
        self.steps += 1
        # Check the horizon
        if self.steps > self.horizon:
            self.gameOver = True
        # Get new state and reward
        self.lastObs = self.getObservation(copy = False)
        self.lastReward = self.calculateReward(self.lastObs)
        return self.lastObs, self.lastReward, self.gameOver

    def validateAction(self, state, action:int):
        if self.movMode == "8C":
            assert (action > 0) and (action < 10), "Action must be an integer between 1 and 9"
            dx, dy = self.actions[action - 1]
        elif self.movMode == "4C":
            assert (action > 0) and (action < 6), "Action must be an integer between 1 and 5"
            dx, dy = self.actions[self.actions4C[action - 1]]
        self.lastAction = action
        posX, posY = state["agent"]
        # Tentative new position
        posX += dx
        posY += dy
        # Checking the movements be inside the grid
        if (posX < 0) or (posX >= self._w) or (posY < 0) or (posY >= self._h):
            # Is not inside the grid, this does nothing
            return state["agent"]
        # Checking if the movement gets it to an obstacle
        elif self.grid[posX, posY] == self.OBST:
            # Returns the same position as before
            return state["agent"]
        else:
            # No obstacle the new position is returned
            return posX, posY
    
    def calculateReward(self, state):
        # For each movement
        reward = -1 
        cellAgent = self.grid[state["agent"]]
        if cellAgent == self.VORTEX:
            # The agent has enter a vortex.
            reward += - 14
            self.gameOver = True
        elif cellAgent == self.GOAL:
            reward += 11
            self.gameOver = True
        return reward 

    def getObservation(self, copy:bool = True):
        if copy:
            return {"agent":(self.posX, self.posY), 
                    "grid": np.copy(self.grid)}
        else:
            return {"agent":(self.posX, self.posY), 
                    "grid": self.grid}

    def render(self, values=None, policy=None):
        # Suboptimal but simple to understand graphics for the environment
        fig = plt.figure(figsize=(self._w * self.GRAPHSCALE, self._h * self.GRAPHSCALE), clear = True)
        self.frame[:,:] = self.EMPTYC
        for i in range(self._w):
            for j in range(self._h):
                cell = self.grid[i,j]
                ni, nj = self.CELLSIZE * i, self.CELLSIZE * j
                f = self.frame[ni:ni+self.CELLSIZE,nj:nj+self.CELLSIZE]
                if cell == self.OBST:
                    f[:,:,:] = self.OBSTC
                elif cell == self.VORTEX:
                    f[self.VORTEXD,:] = self.VORTEXC
                elif cell == self.GOAL:
                    f[self.GOALD,:] = self.GOALC
                if values is not None:
                    plt.text(nj + 1.5, ni + 1.5, str(np.round(values[i,j], 2)),
                             horizontalalignment='center',
                             verticalalignment='center',)
                if policy is not None and (cell == 0):
                    action = policy.getAction((i,j)) - 1
                    if self.movMode == "4C":
                        action = self.actions4C[action]
                    dx, dy = self.actions[action]
                    plt.arrow(nj + 1.5, ni + 1.5, 1.5 * dy, 1.5 * dx, width=0.2, color=self.POLICYC)
        ni, nj = self.posX * self.CELLSIZE, self.posY * self.CELLSIZE
        f = self.frame[ni:ni+self.CELLSIZE,nj:nj+self.CELLSIZE,:] 
        f[self.AGENTD,:] = self.AGENTC
        plt.title("GridWorld {}x{} Action {} Reward {}".format(self._w, self._h, 
                                                               self.lastAction, 
                                                               self.lastReward))
        plt.imshow(self.frame)
        plt.axis("off")

    def updateGrid(self):
        pass

    @property
    def actionSpace(self):
        return self.__actionSpace

    @property
    def observationSpace(self):
        return self._obsSpace

    def transProb(self, state, action):
        # Deterministic Environment
        state = self.validateAction(state, action)
        return [1], [state]

    def isTerminal(self, state):
        if isinstance(state, dict):
            cellAgent = self.grid[state["agent"]]
        else:
            cellAgent = self.grid[state]
        if cellAgent == self.VORTEX or cellAgent == self.GOAL:
            return True
        else:
            return False

    @property
    def shape(self):
        return self.grid.shape

class stochasticGridWorld(gridWorld):
    """
    A modification to the GridWorld to add moving vortex with random directions.
    This movements follow the same type of movement as the agent.

    Parameters
    ----------
    width: int
        First dimension of the grid
    height: int
        Second dimension of the grid
    initPos: tuple of int
        Initial position of the agent.
    goal: tuple of int
        Position of the first goal to create the grid.
        One can add more goals later if required.
    movement: str
        Default 8C. Refer to step method.
    horizon: int
        Default 10**6. Number of steps to run the environment before it
        terminates.

    """
    def __init__(self, width:int, height:int, initPos:tuple, goal:tuple, movement:str = "4C", horizon:int = 10**6):
        super().__init__(width, height, initPos, goal, movement, horizon)
        self.vortexProb = []

    def addVortex(self, *vortex):
        """
        Add a stochastic atraction vortex on the grid
        Parameters
        ---------
        vortex: tuple
            A tuple with the form (x,y,p). x and y are integers 
            which cointain the initial position to put a new vortex.
            While p is a float in [0,1) that the vortex will attract the
            agent to it even if it's not leading to it.
        """
        for v in vortex:
            assert len(v) == 3, "The tuple must cointain two integers as position and third float number to express the probability"
            self.validateTuple(v[:2])
            self.vortex += [v[:2]]
            p = v[2]
            assert (p >= 0) and (p < 1), "The probability; third item on the tuple needs to be between 0 and 1"
            self.vortexProb += [v[2]]
    
    def transProb(self, state, action):
        # Local function
        def nearby(pos:tuple, vortex:tuple, diag:bool):
            d1 = abs(pos[0] - vortex[0])
            d2 = abs(pos[1] - vortex[1])
            if (d1 <= 1) and (d2 <= 1) and (diag == True):
                return True
            elif ((d1 == 1 and d2 == 0) or (d1 == 0 and d2 == 1)) and (diag == False):
                return True
            else:
                return False
        # Checking state type
        if isinstance(state, dict):
            agent = state["agent"]
        else:
            agent = state
        # Init
        states = []
        probs = []
        # Check if the agent is nearby 1 cell of the effect of the vortex
        for v, p in zip(self.vortex, self.vortexProb):
            if nearby(agent, v, True if self.movMode == "8C" else False):
                states += [v]
                probs += [p]
        # Add the action state
        states += [self.validateAction(state, action)]
        n = len(states) - 1
        if n == 0:
            probs = [1]
        else:
            probs += [n - sum(probs)]
            # Normalize the probabilities
            probs = np.array(probs, dtype=np.float32)
            probs = probs / n 
        return probs, states