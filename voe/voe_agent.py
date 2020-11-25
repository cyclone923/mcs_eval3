import random


class VoeAgent:

    def __init__(self, controller, level):
        self.controller = controller
        self.level = level


    def run_scene(self, config):
        self.controller.start_scene(config)
        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])
            self.controller.make_step_prediction(
                choice=random_choice(), confidence=random_confidence(), violations_xy_list=random_voe_list(),
                heatmap_img=step_output.image_list[0], internal_state={}
            )
            if step_output is None:
                break
        self.controller.end_scene(choice=random_choice(), confidence=random_confidence())


def random_float_with_range(low, high):
    assert high > low
    return low + random.random() * (high - low)

def random_choice():
    choice = ["plausible", "implausible"]
    return random.choice(choice)

def random_confidence():
    return random.random()

def random_voe_list():
    n_point = random.randint(0, 9)
    return [{'x': random_float_with_range(0, 600), 'y': random_float_with_range(0, 400)} for _ in range(n_point)]



