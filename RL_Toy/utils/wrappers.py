from gym import Wrapper

class AtariRenderWrapper(Wrapper):
    def render(self, mode="human", **kwargs):
        return self.unwrapped.render()