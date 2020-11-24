import os
import pickle
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
from vision.generateData.save_tool import SaveTool

imgSaver = SaveTool()
cfg = edict()

"""
todo:: modification::
    wall to be FG objects,
       -- left|right|front|back wall to be different objects
    other objects, like chair, desk, trophy, etc.
       -- their real categories are not important
       -- identify trophy | box | others

    panoptic segmentation:
    In summary: FG instance categories: wall | tropy | box | others
                BG semantic: others | ceiling | floor
"""


cfg.bg_classes      = ['floor', 'ceiling', 'wall']
cfg.ignore_fg       = ['shelf']
cfg.object_classes  = ['trophy', 'box', 'occluder_pole', 'occluder_wall']
#object_list = ['changing table', 'duck', 'drawer', 'box', 'bowl', \
#               'sofa chair', 'pacifier', 'number block cube', 'crayon', 'ball', \
#               'blank block cube', 'chair', 'plate', 'sofa', 'stool', \
#               'racecar', 'blank block cylinder', 'cup', 'apple', 'table']
cfg.sem_lut    = {'others_bg': 0}
for ele in cfg.bg_classes:
    cfg.sem_lut[ele] = len(cfg.sem_lut.keys())

cfg.sem_lut['others_fg'] = len(cfg.sem_lut.keys())
for ele in cfg.object_classes:
    cfg.sem_lut[ele] = len(cfg.sem_lut.keys())



def parse_label_info(mask_clrI, uuid_list, shape_list, result_dir='', sname='-dummy'):
    """
    @Param: mask_clrI -- mask in RGB mode, [ht, wd, 3]
            uuid_list -- list of object information, that specify by uuid, default to be BG
            shape_list -- list of object information, that specify by shape, default to be FG
    @Output: save instI and semI as 1 channel image with color-map
    """
    maskI = mask_clrI[..., 0]*1e6 + mask_clrI[..., 1]*1e3 + mask_clrI[..., 2]
    clr_vals = np.unique(maskI)

    # go through uuid list to find fg and bg. default to be BG
    inst_id = 1
    rpl_inst_dict, rpl_sem_dict = dict(), dict()
    for i in uuid_list:
        clr = i.color['r']*1e6 + i.color['g']*1e3 + i.color['b']
        if clr not in clr_vals:
            rpl_sem_dict[clr], rpl_inst_dict[clr] = cfg.sem_lut['others_bg'], 0
        else:
            rpl_sem_dict[clr] = cfg.sem_lut['others_bg']
            # check for FG objects in priority
            for key in (cfg.object_classes + cfg.bg_classes):
                if key in i.uuid:
                    rpl_sem_dict[clr] = cfg.sem_lut[key]
                    break
            if any(s in i.uuid for s in cfg.object_classes):
                rpl_inst_dict[clr], inst_id = inst_id, inst_id+1
            else:
                rpl_inst_dict[clr] = 0

    # go through uuid list to find fg and bg. default to be FG
    for i in shape_list:
        if i.color['r'] is None:
            continue
        clr = i.color['r']*1e6 + i.color['g']*1e3 + i.color['b']
        if clr not in clr_vals or i.shape in cfg.ignore_fg:
            rpl_sem_dict[clr], rpl_inst_dict[clr] = cfg.sem_lut['others_bg'], 0
        else:
            rpl_inst_dict[clr], inst_id = inst_id, inst_id+1
            if i.shape in cfg.object_classes:
                rpl_sem_dict[clr] = cfg.sem_lut[i.shape]
            else:
                rpl_sem_dict[clr] = cfg.sem_lut['others_fg']

    # replace value to generate instI and semI. And save to disk
    instI = np.vectorize(rpl_inst_dict.get)(maskI)
    semI = np.vectorize(rpl_sem_dict.get)(maskI)

    imgSaver.save_single_pilImage_gray(instI, 'label',
                                       save_path=osp.join(result_dir, 'inst'+sname+'.png'))

    imgSaver.save_single_pilImage_gray(semI, 'label',
                                       save_path=osp.join(result_dir, 'cls'+sname+'.png'))


def save_depth_image(depthI, result_dir='', sname='-dummy'):
    imgSaver.save_single_pilImage_gray(depthI, 'range',
             save_path=osp.join(result_dir, 'depth'+sname+'.png'))



