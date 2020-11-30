"""
Different task implementations that can be defined inside an ai2thor environment
"""

from MCS_exploration.gym_ai2thor.utils import InvalidTaskParams


class BaseTask:
    """
    Base class for other tasks to subclass and create specific reward and reset functions
    """
    def __init__(self, config):
        self.task_config = config
        self.max_episode_length = config.get('max_episode_length', 1000)
        # default reward is negative to encourage the agent to move more
        self.movement_reward = config.get('movement_reward', -0.01)
        self.step_num = 0

    def transition_reward(self, state):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError

    def reset(self):
        """

        :param args, kwargs: Configuration for task initialization
        :return:
        """
        raise NotImplementedError


class ExploreAllObjects(BaseTask):
    """
    This task consists of finding all objects in the enviorment.
    """
    def __init__(self, object_id_list, **kwargs):
        super().__init__(kwargs)
        self.target_objects = object_id_list
        self.discoverd = set()

    def transition_reward(self, state):
        reward, done = self.movement_reward, False
        for obj in state.object_list:
            assert obj.uuid in self.target_objects
            if obj.visible and obj.uuid not in self.discoverd:
                self.discoverd.add(obj.uuid)
                x, y, z = obj.direction['x'], obj.direction['y'], obj.direction['z']
                reward += 1

        if self.max_episode_length and self.step_num >= self.max_episode_length or \
                len(self.discoverd) == len(self.target_objects):
            if len(self.discoverd) == len(self.target_objects):
                print("Used {} steps to find all objects".format(self.step_num))
                # reward += len(self.target_objects)
            else:
                print('Totally found objects {}/{} with {} steps'.format(len(self.discoverd), len(self.target_objects),
                                                                     self.step_num))
            done = True
        self.step_num += 1
        return reward, done

    def reset(self):
        self.discoverd = set()
        self.step_num = 0


class NavigationToCoordinate(BaseTask):

    SUCCESS_DISTANCE = 1

    def __init__(self, source, goal, **kwargs):
        super().__init__(kwargs)
        self.goal = goal
        self.prev_distance = self.distance_to_goal(source)

    def distance_to_goal(self, source):
        x1, _, z1 = source
        x2, _, z2 = self.goal
        return ((x1 - x2) ** 2 + (z1 - z2) ** 2) ** 0.5


    def transition_reward(self, state):
        position, action_str = state
        reward, done = self.movement_reward, False
        new_distance = self.distance_to_goal((position['x'], position['y'], position['z']))
        reward += self.prev_distance - new_distance
        self.prev_distance = new_distance
        # if action_str == "Stop":
        if new_distance < self.SUCCESS_DISTANCE:
            print("Success!")
            reward += 10
            done = True
        self.step_num += 1
        if self.step_num == self.max_episode_length:
            done = True
        return reward, done


