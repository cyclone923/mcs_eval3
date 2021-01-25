import matplotlib as m
import matplotlib.pyplot as plt
import numpy as np

DEBUG = False

class GravityAgent:
    def __init__(self, controller, level):
        self.controller = controller
        self.level = level

    def run_scene(self, config, scene_conf):
        if DEBUG:
            print("DEBUG MODE!")

        '''
        # switch plausible scene to implausible
        for o in config["objects"]:
            if o["id"] == "target_object":
                    for step in o["togglePhysics"]:
                        step["stepBegin"] *= 100
        '''

        self.controller.start_scene(config)
        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])
            cam_im = step_output.image_list[0]
            # reverse the channel order: BGR -> RGB
            cam_im = np.array(cam_im)[:,:,::-1]
            mask_im = step_output.object_mask_list[0]
            mask_im = np.array(mask_im)[:,:,::-1]
            # let's check out what we got
            #cv2_show_im(cam_im, mask_im)
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

def cv2_show_im(im, im2=None):
    m.use('TkAgg')
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.1)
    ax1.imshow(im)
    if type(im2) == type(None):
        ax2.imshow(im)
    else:
        ax2.imshow(im2)
    plt.show()