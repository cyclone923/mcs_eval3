import os
import pickle
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
from vision.generateData.save_tool import SaveTool

imgSaver = SaveTool()


def setup_configuration(task='voe', fg_class_en=True):
    """
    @Param: task -- 'voe' | 'interact' | 'combine'
            fg_class_en -- if True, specify the shape of FG objects in detail.
    """
    config = edict()
    config.ignore_fg       = ['shelf']

    # task specification setting
    if task == 'voe':
        config.bg_classes      = []
        if not fg_class_en:
            config.object_classes = ['occluder_pole', 'occluder_wall']
        else:
            config.object_classes  = ['occluder_pole', 'occluder_wall', 'duck', 'cylinder', 'turtle','car', 'sphere', 'cube', 'cone',
                                   'square frustum', 'cylinder']
    elif task == 'interact':
        config.bg_classes      = ['floor', 'ceiling', 'wall']
        if not fg_class_en:
            config.object_classes  = ['trophy', 'box']
        else:
            config.object_classes = ['trophy', 'box', 'changing table', 'drawer', 'shelf',
                                  'blank block cube', 'plate', 'duck', 'sofa chair', 'bowl',
                                  'pacifier', 'crayon', 'number block cube', 'sphere', 'chair',
                                  'sofa', 'stool', 'car', 'blank block cylinder', 'cup', 'apple',
                                  'table', 'crib', 'potted plant']
    else:
        config.bg_classes      = ['floor', 'ceiling', 'wall']
        if not fg_class_en:
            config.object_classes  = ['trophy', 'box', 'occluder_pole', 'occluder_wall']
        else:
            config.object_classes = ['trophy', 'box', 'changing table', 'drawer', 'shelf',
                                  'blank block cube', 'plate', 'duck', 'sofa chair', 'bowl',
                                  'pacifier', 'crayon', 'number block cube', 'sphere', 'chair',
                                  'sofa', 'stool', 'car', 'blank block cylinder', 'cup', 'apple',
                                  'table', 'crib', 'potted plant', 'turtle', 'cone', 'square frustum']

    # construct the semantic look-up-table
    config.sem_lut    = {'others_bg': 0}
    for ele in config.bg_classes:
        config.sem_lut[ele] = len(config.sem_lut.keys())

    config.sem_lut['others_fg'] = len(config.sem_lut.keys())
    for ele in config.object_classes:
        config.sem_lut[ele] = len(config.sem_lut.keys())

    return config


def parse_label_info(cfg, mask_clrI, uuid_list, shape_list, result_dir='', sname='-dummy'):
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

    # imgSaver.save_single_pilImage_gray(semI, 'label',
    #                                    save_path=osp.join(result_dir, 'cls'+sname+'.png'))


def save_depth_image(depthI, result_dir='', sname='-dummy'):
    imgSaver.save_single_pilImage_gray(depthI, 'range',
             save_path=osp.join(result_dir, 'depth'+sname+'.png'))



