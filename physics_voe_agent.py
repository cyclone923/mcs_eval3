from physicsvoe.data.data_gen import convert_output
from physicsvoe import framewisevoe

DEFAULT_CAMERA = {'vfov': 42.5, 'pos': [0, 1.5, -4.5]}

class VoeAgent:
    def __init__(self, controller, level):
        self.controller = controller
        self.level = level
        self.detector = \
            framewisevoe.FramewiseVOE(min_hist_count=3, max_hist_count=8,
                                      dist_thresh=0.5)

    def run_scene(self, config):
        self.controller.start_scene(config)
        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])
            self.calc_voe(step_output, i)
            self.controller.make_step_prediction(
                choice=None, confidence=None, violations_xy_list=None,
                heatmap_img=None, internal_state={}
            )
            if step_output is None:
                break
        self.controller.end_scene(choice=None, confidence=None)

    def calc_voe(self, step_output, frame_num):
        # TODO: calculate object masks
        # TODO: calculate tracking info
        # HACK: Just use oracle masks
        frame = convert_output(step_output)
        depth = frame.depth_mask
        masks = frame.obj_mask
        obj_ids, obj_pos, obj_present = framewisevoe.calc_world_pos(depth, masks, DEFAULT_CAMERA)
        # Infer positions from history
        viols = self.detector.detect(frame_num, obj_pos, obj_ids)
        voe_hmap = framewisevoe.make_voe_heatmap(viols, masks)
        framewisevoe.output_voe(viols)
        framewisevoe.show_scene(frame_num, depth, voe_hmap)
        # Update tracker
        self.detector.record_obs(frame_num, obj_ids, obj_pos, obj_present)
