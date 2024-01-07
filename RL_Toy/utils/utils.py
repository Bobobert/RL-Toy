from typing import Union
import gif
import time
from IPython.display import display, Image
from RL_Toy.base.const import *
from pathlib import Path

render = lambda e : plt.imshow(e.render(mode = 'rgb_array')) 

@gif.frame
def frame(e):
    plt.imshow(e.render(mode = "rgb_array"))

def timeFormatedS() -> str:
    return time.strftime("%H-%M-%S_%d-%b-%y", time.gmtime())

def playGif(src: Union[str, Path]):
    with open(src,'rb') as file_:
        display(Image(file_.read()))

def saveGif(frames, path: Union[str, Path], fps: int = 24):
    duration_between = int(1000 / max(1, fps))
    if isinstance(path, str):
        path = Path.cwd() / path
    gif.save(frames, str(path), duration=duration_between)
    return path
    
def runEnv(env, steps:int, name:str = "lrun", fps: int = 24):
    """
    Run random steps in the environment
    """
    name = name + ".gif"
    frames = []
    totR, epR, eps = 0, 0, 1
    env.reset()
    for _ in range(steps):
        frames.append(frame(env))
        _, reward, done, _= env.step(env.action_space.sample())
        epR += reward
        if done: 
            env.reset()
            eps += 1
            totR += epR
            epR = 0
    totR = totR / eps
    gif_path = saveGif(frames, name, fps=fps)
    print(f"Last run accumulate reward {epR}\nMean accumulate Reward {totR:.2f}\nEpisodes {eps}")
    playGif(gif_path)

def runPolicy(env, policy, steps:int, name:str = None, fps: int = 24):
    """
    Runs, generates and displays a gif in the colab notebook for gym classic control
    environments. Others like ALE don't need this method to display.

    parameters
    ----------
    env: gym environment
        The environment from the gym api
    policy: python object
        An object that works as the policy for said environment. Should
        accept the methods 
        - .getAction(obs)
        - .test as bool
    steps: integer
        number of steps to run the policy on
    name: string
        name for the gif to be named after
    fps: int, default 24
        Frames per second for generated GIF
    """
    name = name + ".gif" if name is not None else "runPolicy {}.gif".format(timeFormatedS())
    frames = []
    policy.test = True
    totR, epR, eps = 0, 0, 1
    obs = env.reset()
    for _ in range(steps):
        frames.append(frame(env))
        action = policy.getAction(obs)
        obs, reward, done, _ = env.step(action)
        epR += reward
        if done: 
            obs = env.reset()
            eps += 1
            totR += epR
            epR = 0
    totR = totR / eps
    policy.test = False
    # Creates .gif
    gifPath = saveGif(frames, name, fps=fps)
    # Prints output
    print(f"Last run accumulate reward {epR}\nMean accumulate Reward {totR:.2f}\nEpisodes {eps}")
    # Displays gif
    playGif(gifPath)
    
def runAgent(agent, steps:int, name: str = None, fps: int = 24):
    """
    Runs, generates and displays a gif in the colab notebook for gym classic control
    environments. Others like ALE don't need this method to display.

    parameters
    ----------
    agent: Agent class
        Some agent class method
    steps: integer
        number of steps to run the policy on
    name: string
        name for the gif to be named after
    fps: int, default 24
        Frames per second for generated GIF
    """
    env = agent.env_test if agent.env_test is not None else agent.env
    policy = agent.policy
    procObs = agent.processObs
    
    name = name + ".gif" if name is not None else "runPolicy {}.gif".format(timeFormatedS())
    frames = []
    policy.test = True
    totR, epR, eps = 0, 0, 1
    obs = env.reset()
    for _ in range(steps):
        frames.append(frame(env))
        state = procObs(obs)
        action = policy.getAction(state)
        obs, reward, done, _ = env.step(action)
        epR += reward
        if done: 
            obs = env.reset()
            eps += 1
            totR += epR
            epR = 0
    totR = totR / eps
    policy.test = False
    # Creates .gif
    gifPath = saveGif(frames, name, fps=fps)
    # Prints output
    print(f"Last run accumulate reward {epR}\nMean accumulate Reward {totR:.2f}\nEpisodes {eps}")
    # Displays gif
    playGif(gifPath)