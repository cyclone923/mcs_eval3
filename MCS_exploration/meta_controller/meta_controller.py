from MCS_exploration import sequence_generator,main #import SequenceGenerator

MAX_REACH_DISTANCE = 1

def get_goal(goal_string):
    goal = goal_string.split("|")
    _, goal_x, goal_y, goal_z = goal
    goal_x = float(goal_x)
    goal_y = float(goal_y)
    goal_z = float(goal_z)
    return (goal_x, goal_y, goal_z)

class MetaController:
    def __init__(self, env, level):
        self.env = env
        self.obstacles = {}
        self.sequence_generator_object = sequence_generator.SequenceGenerator(None, self.env.controller, level)

    def execute(self):
        scene_config = main.explore_scene(self.sequence_generator_object, self.env.step_output)#'retrieval-', '0001'

        return True
