from typing import Any, Tuple
from gym import Wrapper

class AtariRenderWrapper(Wrapper):
    def render(self, mode="human", **kwargs):
        return self.unwrapped.render()
    
class StepCompatible(Wrapper):
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        result = super().step(action)
        if len(result) > 4:
            return result[0], result[1], result[2], result[4]
        return result
    