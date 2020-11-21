import math

class McsWrapper:

    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.reset()

    def step(self, **kwargs):
        self.env.step(**kwargs)

    @property
    def step_output(self):
        return self.env.step_output

    @property
    def agent_x(self):
        return self.env.step_output.position['x']

    @property
    def agent_y(self):
        return self.env.step_output.position['y']

    @property
    def agent_z(self):
        return self.env.step_output.position['z']

    @property
    def agent_rotation(self):
        return self.env.step_output.rotation

    @property
    def agent_head_tilt(self):
        return self.env.step_output.head_tilt

    @property
    def agent_fov(self):
        return self.env.step_output.camera_field_of_view

    @property
    def agent_fov_radian(self):
        return self.agent_fov * math.pi / 180

    @property
    def depth_map(self):
        return self.env.step_output.depth_map_list[0]

    @property
    def image_map(self):
        return self.env.step_output.image_list[0]


