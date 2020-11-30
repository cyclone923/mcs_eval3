
class McsWrapper:

    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

    def step(self, **kwargs):
        self.env.step(**kwargs)

    @property
    def step_output(self):
        return self.env.step_output

