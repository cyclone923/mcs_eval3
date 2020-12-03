from physicsvoe.data.data_gen import convert_output
from physicsvoe import framewisevoe
from tracker import track
from pathlib import Path
from PIL import Image

import numpy as np

DEFAULT_CAMERA = {'vfov': 42.5, 'pos': [0, 1.5, -4.5]}

class VoeAgent:
    def __init__(self, controller, level):
        self.controller = controller
        self.level = level

    def run_scene(self, config):
        name = Path(config['name']).name
        Path(name).mkdir()
        print(name)
        self.track_info = {}
        self.detector = \
            framewisevoe.FramewiseVOE(min_hist_count=3, max_hist_count=8,
                                      dist_thresh=0.5)
        self.controller.start_scene(config)
        scene_voe_detected = False
        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])
            voe_detected, voe_heatmap = self.calc_voe(step_output, i, name)
            scene_voe_detected = scene_voe_detected or voe_detected
            choice = plausible_str(voe_detected)
            assert choice in config['goal']['metadata']['choose'] # Sanity check
            self.controller.make_step_prediction(
                choice=choice, confidence=1.0, violations_xy_list=None,
                heatmap_img=voe_heatmap)
            if step_output is None:
                break
        self.controller.end_scene(choice=plausible_str(scene_voe_detected), confidence=1.0)

    def calc_voe(self, step_output, frame_num, scene_name):
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
        framewisevoe.show_scene(scene_name, frame_num, frame.depth_mask, tracked_masks, voe_hmap)
        # Update tracker
        self.detector.record_obs(frame_num, obj_ids, obj_pos, obj_present)
        # Output results
        voe_detected = viols is not None and len(viols) > 0
        voe_hmap_img = Image.fromarray(voe_hmap)
        return voe_detected, voe_hmap_img

def squash_masks(ref, mask_l, ids):
    flat_mask = np.ones_like(ref) * -1
    for m, id_ in zip(mask_l, ids):
        flat_mask[m] = id_
    return flat_mask

def plausible_str(violation_detected):
    return 'implausible' if violation_detected else 'plausible'
