from physicsvoe.data.data_gen import convert_output
from physicsvoe import framewisevoe
from tracker import track

import numpy as np

DEFAULT_CAMERA = {'vfov': 42.5, 'pos': [0, 1.5, -4.5]}

class VoeAgent:
    def __init__(self, controller, level):
        self.controller = controller
        self.level = level

    def run_scene(self, config):
        self.track_info = {}
        self.detector = \
            framewisevoe.FramewiseVOE(min_hist_count=3, max_hist_count=8,
                                      dist_thresh=0.5)
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
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

    def calc_voe(self, step_output, frame_num):
        frame = convert_output(step_output)
        # TODO: calculate object masks
        masks = frame.obj_mask
        # Calculate tracking info
        self.track_info = track.track_objects(masks, self.track_info)
        obj_ids = list(range(self.track_info['object_index']))
        masks_list = [self.track_info['objects'][i]['mask'] for i in obj_ids]
        tracked_masks = squash_masks(frame.depth_mask, masks_list, obj_ids)
        # Calculate object level info from masks
        obj_ids, obj_pos, obj_present = \
            framewisevoe.calc_world_pos(frame.depth_mask, tracked_masks, frame.camera)
        # Calculate violations
        viols = self.detector.detect(frame_num, obj_pos, obj_ids, frame.depth_mask, frame.camera)
        voe_hmap = framewisevoe.make_voe_heatmap(viols, tracked_masks)
        # Output violations
        framewisevoe.output_voe(viols)
        framewisevoe.show_scene(frame_num, frame.depth_mask, tracked_masks, voe_hmap)
        # Update tracker
        self.detector.record_obs(frame_num, obj_ids, obj_pos, obj_present)

def squash_masks(ref, mask_l, ids):
    flat_mask = np.ones_like(ref) * -1
    for m, id_ in zip(mask_l, ids):
        flat_mask[m] = id_
    return flat_mask
