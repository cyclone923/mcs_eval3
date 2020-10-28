from tasks.point_goal_navigation.navigator import NavigatorResNet
from gym_ai2thor.envs.mcs_wrapper import McsWrapper
import machine_common_sense
import numpy as np


class McsNavWrapper(McsWrapper):

    ABS_ROTATION = 10 # same as settings in habitat, right is positive
    ABS_MOVE = 0.5

    def __init__(self, env):
        super().__init__(env)
        self.action_names = ["Stop", "MoveAhead", "RotateLeft", "RotateRight"]# order matters

    def step(self, action, epsd_collector=None, frame_colletor=None):
        assert action in self.action_names
        if action.startswith('Rotate'):
            rotation = -self.ABS_ROTATION if action == 'RotateLeft' else self.ABS_ROTATION
            super().step(action='RotateLook', rotation=rotation)
        elif action == 'MoveAhead':
            super().step(action=action, amount=self.ABS_MOVE)
        else:
            assert action == 'Stop'
            # raise AttributeError('Navigator Should End Before Stop')

        return self.step_output

    def set_look_dir(self, rotation_in_all=0):
        n = int(abs(rotation_in_all) // 10)
        if rotation_in_all > 0:
            for _ in range(n):
                super().step(action="RotateRight")
        else:
            for _ in range(n):
                super().step(action="RotateLeft")

    def micro_move(self, env, goal):
        current_x, current_z = env.step_output.position['x'], env.step_output.position['z']
        goal_x, _, goal_z = goal
        delta_x, delta_z = goal_x - current_x, goal_z - current_z
        amount = (delta_x ** 2 + delta_z ** 2) ** 0.5
        rotate = NavigatorResNet.get_polar_direction(goal, env.step_output) * 360 / (2 * np.pi)
        super().step(action='RotateLook', rotation=-rotate)
        while amount > machine_common_sense.mcs_controller_ai2thor.MAX_MOVE_DISTANCE:
            super().step(action='MoveAhead', amount=1)
            amount -= machine_common_sense.mcs_controller_ai2thor.MAX_MOVE_DISTANCE
        super().step(action='MoveAhead', amount=amount / machine_common_sense.mcs_controller_ai2thor.MAX_MOVE_DISTANCE)












