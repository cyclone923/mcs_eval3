from exploration.mcs_env.mcs_wrapper import McsWrapper
import math

INIT_X = 0
INIT_Y = 0.762
INIT_Z = 0
INIT_R = 0
INIT_H = 0

MOVE_DISTANCE = 0.5
ABS_MOVE = MOVE_DISTANCE

ABS_ROTATE = 10

def similar(a, b, eps=1e-2):
    return abs(a - b) < eps


class DeadReckoning:

    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.r = None


    def reset(self, env=None):
        if env:
            self.x = env.step_output.position['x']
            self.y = env.step_output.position['y']
            self.z = env.step_output.position['z']
            self.r = env.step_output.rotation
            self.h = env.step_output.head_tilt
        else:
            self.x = INIT_X
            self.y = INIT_Y
            self.z = INIT_Z
            self.r = INIT_R
            self.h = INIT_H

    def update(self, **kwargs):
        action = kwargs['action']
        if action == "MoveAhead":
            r_radian = math.radians(self.r)
            dx = MOVE_DISTANCE * math.sin(r_radian)
            dz = MOVE_DISTANCE * math.cos(r_radian)
            self.x += dx
            self.z += dz
        else:
            if "Rotate" in action:
                if action == "RotateRight":
                    self.r += ABS_ROTATE
                elif action == "RotateLeft":
                    self.r -= ABS_ROTATE
                if self.r >= 360:
                    self.r -= 360
                if self.r < 0:
                    self.r += 360
            elif "Look" in action:
                if action == "LookDown":
                    self.h += ABS_ROTATE
                elif action == "LookUp":
                    self.h -= ABS_ROTATE

class DeadReckoningMcsWrapper(McsWrapper):
    def __init__(self, env, level):
        McsWrapper.__init__(self, env)
        self.level = level

        self.dead_reckonging = None
        if self.level != 'oracle':
            self.dead_reckonging = DeadReckoning()

    def reset(self):
        McsWrapper.reset(self)
        if self.dead_reckonging:
            self.dead_reckonging.reset()

    def step(self, **kwargs):
        McsWrapper.step(self, **kwargs)
        if self.step_output.return_status != "SUCCESSFUL":
            print(self.step_output.return_status)
            exit(0)
        else:
            if self.dead_reckonging:
                self.dead_reckonging.update(**kwargs)
        # action = kwargs['action']
        # if not similar(self.agent_x, super().agent_x):
        #     print(action)
        #     print("x", self.agent_x, super().agent_x)
        #     exit(0)
        # if not similar(self.agent_y, super().agent_y):
        #     print(action)
        #     print("y", self.agent_y, super().agent_y)
        #     exit(0)
        # if not similar(self.agent_z, super().agent_z):
        #     print(action)
        #     print("z", self.agent_z, super().agent_z)
        #     exit(0)
        # if not similar(self.agent_head_tilt, super().agent_head_tilt):
        #     print(action)
        #     print("h", self.agent_head_tilt, super().agent_head_tilt)
        #     exit(0)
        # if not similar(self.agent_rotation, super().agent_rotation):
        #     print(action)
        #     print("r", self.agent_rotation, super().agent_rotation)
        #     exit(0)

    @property
    def agent_x(self):
        if self.level != 'oracle':
            agent_x = self.dead_reckonging.x
        else:
            agent_x = McsWrapper.agent_x.fget(self)
        return agent_x

    @property
    def agent_y(self):
        if self.level != 'oracle':
            agent_y = self.dead_reckonging.y
        else:
            agent_y = McsWrapper.agent_y.fget(self)
        return agent_y

    @property
    def agent_z(self):
        if self.level != 'oracle':
            agent_z = self.dead_reckonging.z
        else:
            agent_z = McsWrapper.agent_z.fget(self)
        return agent_z

    @property
    def agent_rotation(self):
        if self.level != 'oracle':
            agent_r = self.dead_reckonging.r
        else:
            agent_r = McsWrapper.agent_rotation.fget(self)
        return agent_r

    @property
    def agent_rotation_radian(self):
        return math.radians(self.agent_rotation)

    @property
    def agent_head_tilt_radian(self):
        return math.radians(self.agent_head_tilt)







