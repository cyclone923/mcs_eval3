from .base_dataset import *
from .base_config import MEANS, STD, COLORS, activation_func
from .base_config import overwrite_args_from_json
from .base_config import overwrite_params_from_json


def dataset_specific_import(dataName):
    '''
    import dataset related functions
    '''

    from .config_mcsVideo3_inter import cfg, set_dataset, set_cfg
    # if 'mcsvideo3_inter' in dataName:
    #     from .config_mcsVideo3_inter import cfg, set_dataset, set_cfg
    # elif 'mcsvideo3_voe' in dataName:
    #     from .config_mcsVideo3_voe import cfg, set_dataset, set_cfg
    # elif 'voe' in dataName:
    #     from vision.instSeg.data.config_mcsVideo_voe import cfg, set_dataset, set_cfg
    # else: #if 'interact' in dataName:
    #     from vision.instSeg.data.config_mcsVideo_inter import cfg, set_dataset, set_cfg

    from data.mcsVideo import MCSVIDEODetection as DataSet

    return DataSet, cfg, set_dataset, set_cfg

