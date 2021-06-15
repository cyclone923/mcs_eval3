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
    def __init__(self, env, level, collector=None):
        self.env = env
        self.obstacles = {}
        self.collector = collector
        self.sequence_generator_object = sequence_generator.SequenceGenerator(None, self.env.controller, level, frame_collector=collector)

    def execute(self):
        scene_config = main.explore_scene(self.sequence_generator_object, self.env.step_output, frame_collector=self.collector, scene_type='retrieval-')#, self.env.collector.scene_number)

        return True
