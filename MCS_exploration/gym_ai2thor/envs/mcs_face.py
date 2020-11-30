from MCS_exploration.gym_ai2thor.envs.mcs_wrapper import McsWrapper
from tasks.point_goal_navigation.navigator import NavigatorResNet
import numpy as np

CAMERA_HIGHT = 2

class McsFaceWrapper(McsWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.action_names = [
            "MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Stop"
        ]

    def step(self, action):
        assert action in self.action_names
        if action == "LookUp":
            super().step(action="LookUp")
        elif action == "LookDown":
            super().step(action="LookDown")
        elif action == "RotateLeft":
            super().step(action="RotateLeft")
        elif action == "RotateRight":
            super().step(action="RotateRight")
        elif action == "MoveAhead":
            super().step(action="MoveAhead")
        elif action == "MoveBack":
            super().step(action="MoveBack")
        elif action == "MoveLeft":
            super().step(action="MoveLeft")
        elif action == "MoveRight":
            super().step(action="MoveRight")

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








