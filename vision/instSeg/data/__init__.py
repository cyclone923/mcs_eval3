from .base_config import MEANS, STD, COLORS, activation_func
from .base_config import overwrite_from_json_config
from .base_dataset import *


def dataset_specific_import(dataName):
    '''
    import dataset related functions
    '''
    if 'mcsvideo' in dataName:
        if 'mcsvideo3' in dataName:
            from data.config_mcsVideo3_inter import cfg, set_cfg
        elif 'voe' in dataName:
            from data.config_mcsVideo_voe import cfg, set_cfg
        else: #if 'interact' in dataName:
            from data.config_mcsVideo_inter import cfg, set_cfg
    else:
        print('please specify a supported dtaset!')
        exit(-1)

    return cfg, set_cfg

