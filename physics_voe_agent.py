from physicsvoe.data.data_gen import convert_output
from physicsvoe import framewisevoe, occlude
from physicsvoe.timer import Timer
from physicsvoe.data.types import make_camera

from tracker import track, e3_appearence as appearence, filter_masks, appearance as e4_appearance
import visionmodule.inference as vision

from pathlib import Path
from PIL import Image
import numpy as np
import torch
import pickle

from rich.console import Console

console = Console()

APP_MODEL_PATH = './tracker/model.p'
VISION_MODEL_PATH = './visionmodule/dvis_resnet50_mc_voe.pth'
DEBUG = False

class VoeAgent:
    def __init__(self, controller, level, out_prefix=None, type='csrt'):
        self.controller = controller
        self.level = level
        if DEBUG:
            self.prefix = out_prefix
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Create appearace model, load its (hardcoded) pretrained weights
        self.app_model = e4_appearance.AppearanceMatchModel(type)
        if type == 'sift':
            self.app_model.modeler.eval()
        # self.app_model.load_state_dict(torch.load(APP_MODEL_PATH, map_location=torch.device(self.device)))
        # self.app_model = self.app_model.to(self.device).eval()
        # The full vision model is used for object mask prediction, so it
        # isn't used for the level2/oracle levels (where masks are provided).
        if self.level == 'level1':
            self.visionmodel = vision.MaskAndClassPredictor(dataset='mcsvideo3_voe',
                                                            config='plus_resnet50_config_depth_MC',
                                                            weights=VISION_MODEL_PATH)

    def run_scene(self, config, desc_name):
        # console.print(desc_name, style='blue bold')
        if DEBUG:
            # Create folder for debug output, named after scene's json file
            folder_name = Path(self.prefix)/Path(Path(desc_name).stem)
            if folder_name.exists():
                # ABORT and don't run the scene if we already have output
                # for this scene. Presumably if we're collecting debug output
                # we don't want to generate the same data twice.
                return None
            folder_name.mkdir(parents=True)
            console.print(folder_name)
        else:
            folder_name = None
        self.track_info = {}
        self.detector = \
            framewisevoe.FramewiseVOE(min_hist_count=3, max_hist_count=8,
                                      dist_thresh=0.5)
        self.controller.start_scene(config)
        scene_voe_detected = False
        all_viols = []
        all_errs = []
        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])
            # console.print(step_output)
            voe_detected, voe_heatmap, voe_xy_list, viols, frame_errs = self.calc_voe(step_output, i, folder_name)
            console.print('VOE' if voe_detected else 'Not VOE', style='green' if voe_detected else 'red')
            console.print('')
            all_viols.append(viols)
            all_errs += frame_errs
            console.print('Violations:', viols)
            console.print('Errors:', frame_errs)
            scene_voe_detected = scene_voe_detected or voe_detected
            console.print('[yellow]Scene-Wide VOE Detected?[/yellow]', scene_voe_detected, style='yellow')
            choice = plausible_str(voe_detected)
            if DEBUG:
                assert choice in config['goal']['metadata']['choose'] # Sanity check
            self.controller.make_step_prediction(
                choice=choice, confidence=1.0, violations_xy_list=voe_xy_list,
                heatmap_img=voe_heatmap)
            if step_output is None:
                break
        self.controller.end_scene(choice=plausible_str(scene_voe_detected), confidence=1.0)
        if DEBUG:
            with open(folder_name/'viols.pkl', 'wb') as fd:
                pickle.dump((all_viols, all_errs), fd)
        return scene_voe_detected

    def calc_voe(self, step_output, frame_num, scene_name=None):
        # Use the most recent depth map and RGB frames from the simulator's
        # output
        depth_map = step_output.depth_map_list[-1]
        rgb_image = step_output.image_list[-1]
        # print(rgb_image)
        # We need to calculate+store some camera properties so that we can
        # project points between screen space into world space
        camera_info = make_camera(step_output)
        # Get the object mask data, depending on level
        if self.level == 'oracle':
            masks = self.oracle_masks(step_output)
        elif self.level == 'level2':
            in_mask = step_output.object_mask_list[-1]
            masks = self.level2_masks(depth_map, rgb_image, in_mask)
        elif self.level == 'level1':
            masks = self.level1_masks(depth_map, rgb_image)
        else:
            raise ValueError(f'Unknown level `{self.level}`')
        # Calculate tracking info using mask history
        # This stores a bunch of state in `self.track_info`
        self.track_info = track.track_objects(masks, self.track_info)

        # The tracking model assigns an ID to each mask that is consistent
        # across frames, and calculates a mask for each object ID.
        # We use `squash_masks` to turn this list of object masks into a single
        # one-hot encoded matrix associating each screen pixel with an object.
        all_obj_ids = list(range(self.track_info['object_index']))
        masks_list = [self.track_info['objects'][i]['mask'] for i in all_obj_ids]
        tracked_masks = squash_masks(depth_map, masks_list, all_obj_ids)
        # Call 'occlusion' model
        # 'Occlusion' really just means that the object is sufficiently
        # obscured that we can't rely on our appearance model or position
        # estimation.
        # Therefore, if an object is determined to be 'occluded' we just ignore
        # all raised VOEs.
        area_hists = {o_id:o_info['area_history'] for o_id, o_info in self.track_info['objects'].items()}
        obj_occluded = occlude.detect_occlusions(depth_map, tracked_masks, all_obj_ids, area_hists)
        console.print('[yellow]Objects occluded?[/yellow]', obj_occluded)

        for o_id in range(0, len(obj_occluded)):
            # print(o_id)
            self.track_info['objects'][o_id]['occluded'] = obj_occluded[o_id]
            # print(self.track_info['objects'])

        # Add appearance model's output to the object tracking info.
        self.track_info['objects'] = self.app_model.appearanceMatch(rgb_image, self.track_info['objects'], self.device, self.level)
        if DEBUG:
            # Generate+output appearance model's thoughts
            img = appearence.draw_bounding_boxes(rgb_image, self.track_info['objects'])
            img = appearence.draw_appearance_bars(img, self.track_info['objects'])
            img.save(scene_name/f'DEBUG_{frame_num:02d}.png')
        
        # Calculate object level info from masks
        # We can use the object masks + depth map + camera data to easily
        # estimate world_space positions for each object in the scene.
        obj_ids, obj_pos, obj_present = \
            framewisevoe.calc_world_pos(depth_map, tracked_masks, camera_info)
        occ_heatmap = framewisevoe.make_occ_heatmap(obj_occluded, obj_ids, tracked_masks)
        # Calculate violations
        det_result = self.detector.detect(frame_num, obj_pos, obj_occluded, obj_ids, depth_map, camera_info)
        if det_result is None:
            # Early return from VOE detector = no VOEs present
            dynamics_viols = []
            all_errs = []
        else:
            dynamics_viols, all_errs = det_result
        console.print('[yellow]Dynamics violations:[/yellow]', dynamics_viols)
        appearance_viols = []
        for o_id, obj_info in self.track_info['objects'].items():
            console.log('Object ID:', o_id)
            o_visible = obj_info['visible']
            o_mismatch = not obj_info['appearance']['match']
            o_occluded = obj_occluded[o_id]
            console.log('Visible?', o_visible)
            console.log('Appearance mismatch?', o_mismatch)
            console.log('Occluded?', o_occluded)
            o_robust_mismatch = o_mismatch and obj_info['appearance']['mismatch_count'] > 3
            console.log('Robust appearance mismatch?', o_robust_mismatch)
            app_viol = o_visible and o_robust_mismatch and not o_occluded
            if app_viol:
                _idx = obj_ids.index(o_id)
                appearance_viols.append(framewisevoe.AppearanceViolation(o_id, obj_pos[_idx]))
        console.print('[yellow]Appearance violations:[/yellow]', appearance_viols)
        # Update tracker
        vis_count = {o_id:o_info['visible_count'] for o_id, o_info in self.track_info['objects'].items()}
        pos_hists = {o_id:o_info['position_history'] for o_id, o_info in self.track_info['objects'].items()}
        obs_viols = self.detector.record_obs(frame_num, obj_ids, obj_pos, obj_present, obj_occluded, vis_count, pos_hists, camera_info)
        console.print('[yellow]Observation violations:[/yellow]', obs_viols)
        # Output violations
        viols = dynamics_viols + obs_viols
        if self.level != 'level1': #Ignore appearance violations in the level1 case
            viols += appearance_viols
        # Create VOE heatmap from the list of violations, for step output
        voe_hmap = framewisevoe.make_voe_heatmap(viols, tracked_masks)
        if DEBUG:
            framewisevoe.output_voe(viols)
            framewisevoe.show_scene(scene_name, frame_num, depth_map, tracked_masks, voe_hmap, occ_heatmap)
        # Output results
        voe_detected = viols is not None and len(viols) > 0
        voe_hmap_img = Image.fromarray(voe_hmap)
        voe_xy_list = [v.xy_pos(camera_info) for v in (viols or [])]
        return voe_detected, voe_hmap_img, voe_xy_list, viols, all_errs

    def oracle_masks(self, step_output):
        frame = convert_output(step_output)
        return frame.obj_mask

    def level2_masks(self, depth_img, rgb_img, mask_img):
        in_mask = np.array(mask_img)
        unique_cols = np.unique(in_mask.reshape(-1, 3), axis=0)
        split_masks = [(in_mask == col).all(axis=-1) for col in unique_cols]
        filter_result = filter_masks.filter_objects(rgb_img, depth_img, split_masks)
        masks = -1 * np.ones(in_mask.shape[:2], dtype=np.int)
        for i, o in enumerate(filter_result['objects']):
            masks[o] = i
        return masks

    def level1_masks(self, depth_img, rgb_img):
        bgr_img = np.array(rgb_img)[:, :, [2, 1, 0]]
        result = self.visionmodel.step(bgr_img, depth_img)
        filter_result = filter_masks.filter_objects_model(rgb_img, depth_img, result)
        masks = -1 * np.ones(np.array(depth_img).shape[:2], dtype=np.int)
        for i, o in enumerate(filter_result['objects']):
            masks[o] = i
        return masks

def squash_masks(ref, mask_l, ids):
    flat_mask = np.ones_like(ref) * -1
    for m, id_ in zip(mask_l, ids):
        flat_mask[m] = id_
    return flat_mask

def prob_to_mask(prob, cutoff, obj_scores):
    obj_pred_class = obj_scores.argmax(-1)
    valid_ids = []
    for mask_id, pred_class in zip(range(cutoff, prob.shape[0]), obj_pred_class):
        if pred_class == 1: #An object!
            valid_ids.append(mask_id+cutoff)
    out_mask = -1 * np.ones(prob.shape[1:], dtype=np.int)
    am = np.argmax(prob, axis=0)
    for out_id, mask_id in enumerate(valid_ids):
        out_mask[am==mask_id-cutoff] = out_id
    return out_mask

def plausible_str(violation_detected):
    return 'implausible' if violation_detected else 'plausible'
