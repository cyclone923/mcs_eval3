import numpy as np

DEBUG = False

class GravityAgent:
    def __init__(self, controller, level):
        self.controller = controller
        self.level = level

    def run_scene(self, config, desc_name):
        if DEBUG:
            print("DEBUG MODE!")
        self.controller.start_scene(config)
        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])
            choice = plausible_str(True)
            voe_xy_list = []
            voe_heatmap = None
            self.controller.make_step_prediction(
                choice=choice, confidence=1.0, violations_xy_list=voe_xy_list,
                heatmap_img=voe_heatmap)
            if step_output is None:
                break
        self.controller.end_scene(choice=choice, confidence=1.0)
        return True

    def calc_voe(self, step_output, frame_num, scene_name=None):
        pass

def plausible_str(violation_detected):
    return 'implausible' if violation_detected else 'plausible'
