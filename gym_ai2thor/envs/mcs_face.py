from gym_ai2thor.envs.mcs_wrapper import McsWrapper
from tasks.point_goal_navigation.navigator import NavigatorResNet
import numpy as np

CAMERA_HIGHT = 2

class McsFaceWrapper(McsWrapper):

    ABS_HEADTILT = 10
    ABS_ROTATION = 10 # same as settings in habitat, right is positive
    ABS_MOVE = 0.1

    def __init__(self, env):
        super().__init__(env)

        self.action_names = [
            "MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Stop"
        ]

    def step(self, action):
        assert action in self.action_names
        if action == "LookUp":
            super().step(action="RotateLook", horizon=-self.ABS_HEADTILT)
        elif action == "LookDown":
            super().step(action="RotateLook", horizon=self.ABS_HEADTILT)
        elif action == "RotateLeft":
            super().step(action="RotateLook", rotation=-self.ABS_ROTATION)
        elif action == "RotateRight":
            super().step(action="RotateLook", rotation=self.ABS_ROTATION)
        elif action == "MoveAhead":
            super().step(action="MoveAhead", amount=self.ABS_MOVE)
        elif action == "MoveBack":
            super().step(action="MoveBack", amount=self.ABS_MOVE)
        elif action == "MoveLeft":
            super().step(action="MoveLeft", amount=self.ABS_MOVE)
        elif action == "MoveRight":
            super().step(action="MoveRight", amount=self.ABS_MOVE)

    def set_look_dir(self, rotation_in_all=0, horizon_in_all=0):
        n = int(abs(rotation_in_all) // 10)
        m = int(abs(horizon_in_all) // 10)
        if rotation_in_all > 0:
            for _ in range(n):
                super().step(action="RotateRight")
        else:
            for _ in range(n):
                super().step(action="RotateLeft")

        if horizon_in_all > 0:
            for _ in range(m):
                super().step(action="LookDown")
        else:
            for _ in range(m):
                super().step(action="LookUp")








