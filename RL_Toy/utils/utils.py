import gif
from IPython.display import display, Image
from RL_Toy.base.const import *

render = lambda e : plt.imshow(e.render(mode = 'rgb_array')) 

@gif.frame
def frame(e):
    plt.imshow(e.render(mode = "rgb_array"))

def repGIF(src: str):
    with open(src,'rb') as file:
        display(Image(file.read()))
        
def runEnv(env, steps:int, name:str = "lrun"):
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
    gif.save(frames, name, duration = steps * 0.1, unit="s", between="startend")
    print("Mean accumulate Reward {:.2f}, episodes {}".format(totR, eps))
    repGIF(name)