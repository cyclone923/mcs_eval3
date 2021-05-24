from pathlib import Path
from .utils import draw_bounding_boxes, draw_appearance_bars, split_obj_masks, get_obj_position, get_mask_box
from physicsvoe.data.data_gen import convert_output
from physicsvoe import occlude
from physicsvoe import framewisevoe, occlude
from physicsvoe.timer import Timer
from physicsvoe.data.types import make_camera
import visionmodule.inference as vision
# from types import ThorFrame, CameraInfo
from .track import track_objects
from PIL import Image
from tracker import filter_masks
import torch
from collections import namedtuple
import numpy as np


from .appearance_wrapper import object_appearance_match


#This is only for testing Siamese appearance. It doesn't track any position/presence voilations 
#Steps:
#1. Run the scene config file
#2. Get the masks and depths of the objects, from the stepoutput 
#3. Get the tracking info history
#4. For every object in the history, if not occluded: test the appearance

DEBUG = True

APP_MODEL_PATH = '/home/gulsh/mcs_opics/tracker/model_iter-3999.pth'
VISION_MODEL_PATH = './visionmodule/dvis_resnet50_mc_voe.pth'
class VoeAgent:
    def __init__(self, controller, level, out_prefix=None):
        self.controller = controller
        self.level = level
        if DEBUG:
            self.prefix = out_prefix
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.app_model = appearence.AppearanceMatchModel()
        # self.app_model.load_state_dict(torch.load(APP_MODEL_PATH, map_location=torch.device('cpu')))
        # self.app_model = self.app_model.to(self.device).eval()
        if self.level == 'level1':
            self.visionmodel = vision.MaskAndClassPredictor(dataset='mcsvideo3_voe',
                                                            config='plus_resnet50_config_depth_MC',
                                                            weights=VISION_MODEL_PATH)


    def run_scene(self, config, desc_name):
        if DEBUG:
            folder_name = Path(self.prefix)/Path(Path(desc_name).stem)
            if folder_name.exists():
                return None
            folder_name.mkdir(parents=True)
            print(folder_name)
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
            voe_detected, voe_heatmap, voe_xy_list, viols, frame_errs = self.calc_voe(step_output, i, folder_name)
            all_viols.append(viols)
            all_errs += frame_errs
            scene_voe_detected = scene_voe_detected or voe_detected
            choice = 'implausible' if scene_voe_detected else 'plausible'
            if DEBUG:
                assert choice in config['goal']['metadata']['choose'] # Sanity check
            self.controller.make_step_prediction(
                choice=choice, confidence=1.0, violations_xy_list=voe_xy_list,
                heatmap_img=voe_heatmap)
            if step_output is None:
                break
        self.controller.end_scene(choice=plausible_str(scene_voe_detected), confidence=1.0)
        # if DEBUG:
        #     with open(folder_name/'viols.pkl', 'wb') as fd:
        #         pickle.dump((all_viols, all_errs), fd)
        print(f'Scene wide VOE:{scene_voe_detected}')
        return scene_voe_detected


    def calc_voe(self, step_output, frame_num, scene_name=None):
        depth_map = step_output.depth_map_list[-1]
        rgb_image = step_output.image_list[-1]
        camera_info = make_camera(step_output)
        if self.level == 'oracle':
            masks = self.oracle_masks(step_output)
        elif self.level == 'level2':
            in_mask = step_output.object_mask_list[-1]
            masks = self.level2_masks(depth_map, rgb_image, in_mask)
        elif self.level == 'level1':
            masks = self.level1_masks(depth_map, rgb_image)
        else:
            raise ValueError(f'Unknown level `{self.level}`')

        #get the tracking info
        print("hello!!! from calc voe")
        self.track_info = track_objects(masks, self.track_info)

        all_obj_ids = list(range(self.track_info['object_index']))
        masks_list = [self.track_info['objects'][i]['mask'] for i in all_obj_ids]
        tracked_masks = squash_masks(depth_map, masks_list, all_obj_ids)
        # Calculate occlusion from masks
        area_hists = {o_id:o_info['area_history'] for o_id, o_info in self.track_info['objects'].items()}
        # pos_hists = {o_id:o_info['position_history'] for o_id, o_info in self.track_info['objects'].items()}
        # print("before: ",pos_hists)
        obj_occluded = occlude.detect_occlusions(depth_map, tracked_masks, all_obj_ids, area_hists)

        self.track_info['objects'] = object_appearance_match(rgb_image, scene_name,frame_num, self.track_info['objects'], self.device, self.level)
        if DEBUG:
            img = draw_bounding_boxes(rgb_image, self.track_info['objects'])
            img.save(scene_name/f'DEBUG_{frame_num:02d}.png')

        obj_ids, obj_pos, obj_present = \
            framewisevoe.calc_world_pos(depth_map, tracked_masks, camera_info)
        occ_heatmap = framewisevoe.make_occ_heatmap(obj_occluded, obj_ids, tracked_masks)
        # Calculate violations
        det_result = self.detector.detect(frame_num, obj_pos, obj_occluded, obj_ids, depth_map, camera_info)
        if det_result is None:
            dynamics_viols = []
            all_errs = []
        else:
            dynamics_viols, all_errs = det_result
        appearance_viols = []

        
        for o_id, obj_info in self.track_info['objects'].items():
            o_visible = obj_info['visible']
            if 'appearance' in obj_info.keys():
                o_mismatch = not obj_info['appearance']['match']
                o_occluded = obj_occluded[o_id]
                o_robust_mismatch = o_mismatch and obj_info['appearance']['mismatch_count'] > 3
                app_viol = o_visible and o_robust_mismatch and not o_occluded
                if app_viol:
                    _idx = obj_ids.index(o_id)
                    appearance_viols.append(framewisevoe.AppearanceViolation(o_id, obj_pos[_idx]))
        # Update tracker
        # console.print('[yellow]Appearance violations:[/yellow]', appearance_viols)
        vis_count = {o_id:o_info['visible_count'] for o_id, o_info in self.track_info['objects'].items()}
        pos_hists = {o_id:o_info['position_history'] for o_id, o_info in self.track_info['objects'].items()}
        obs_viols = self.detector.record_obs(frame_num, obj_ids, obj_pos, obj_present, obj_occluded, vis_count, pos_hists, camera_info)
        # Output violations
        viols = dynamics_viols + obs_viols
        if self.level != 'level1': #Ignore appearance violations in the level1 case
            viols += appearance_viols
        voe_hmap = framewisevoe.make_voe_heatmap(viols, tracked_masks)
        # if DEBUG:
        #     framewisevoe.output_voe(viols)
        #     framewisevoe.show_scene(scene_name, frame_num, depth_map, tracked_masks, voe_hmap, occ_heatmap)
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

def plausible_str(violation_detected):
    return 'implausible' if violation_detected else 'plausible'

def squash_masks(ref, mask_l, ids):
    flat_mask = np.ones_like(ref) * -1
    for m, id_ in zip(mask_l, ids):
        flat_mask[m] = id_
    return flat_mask