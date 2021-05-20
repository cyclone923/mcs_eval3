import os
import numpy as np
from skimage import measure as smeasure

import torch
from torch.nn import functional as F

import sys
sys.path.append('./vision/instSeg')

from vision.instSeg import data
from vision.instSeg.dvis_network import DVIS
from vision.instSeg.utils.augmentations import BaseTransform


def voe_connect_comp_analysis(mask_probs, obj_scores, fg_stCH=1, wronglabel=[6], size_thr=10):
    '''
    @Func: 20201204: this func works as a trick to fix the wrong prediction on VOE scenes that:
                 when there are three occl-poles and occl-walls, it failed to segment moving objects.
                 That, the moving objects are messed together with one pole.
           Sol: when there are 6 objects detected, perform component analysis.

    @Param: mask_probs -- [fg_stCH + N, ht, wd]
            obj_scores -- [N, num_classes]
    @Output: ret_mask_probs -- [fg_stCH + N + K, ht, wd]
             ret_obj_classes -- [N + K, num_classes]
    '''
    def _is_pole(prop):
        y0,x0,y1,x1 = prop.bbox
        if (y1-y0)/(x1-x0) > 5:
            return True
        else:
            return False

    ch, ht, wd = mask_probs.shape
    labelI = mask_probs.argmax(axis=0)

    # on observed wrong lables, perform connected component analysis
    ext_masks, ext_scores = [], []
    for i in wronglabel:
        conn_labelI = smeasure.label(labelI==i)
        props = smeasure.regionprops(conn_labelI)
        if len(props) ==1:
            continue
        else:
            for prop in props:
                # check if it is pole or object
                if _is_pole(prop):
                    obj_scores[i-fg_stCH, :] = np.asarray([0.17, 0.17, 0.5, 0.16])
                elif prop.area > size_thr:
                    flagI = conn_labelI == prop.label
                    ext_masks.append(flagI*mask_probs[i])
                    ext_scores.append([0.17, 0.5, 0.17, 0.16])
                    mask_probs[i][flagI] = 0
                else:
                    continue
    if len(ext_masks)>0:
        ext_masks = np.stack(ext_masks)
        ext_scores = np.asarray(ext_scores)

        mask_probs = np.concatenate([mask_probs, ext_masks], axis=0)
        obj_scores = np.concatenate([obj_scores, ext_scores], axis=0)

    return mask_probs, obj_scores


class MaskAndClassPredictor(object):
    '''
    The class is used to load trained DVIS-MC model and predict panoptic segmentation from raw data (RGB or RGB+D)
    '''
    def __init__(self, dataset='mcsvideo3_inter',
                       config='plus_resnet50_config_depth_MC',
                       weights=None):
        '''
        @Param: dataset -- 'mcsvideo3_inter | mcsvideo3_voe | mcsvideo_inter | mcsvideo_voe'
                config -- check the config files in data for more other configurations.
                weights -- file for loading model weights
        '''
        cfg, set_cfg = data.dataset_specific_import(dataset)
        set_cfg(cfg, config)

        self.dataset_name = dataset
        self.fg_stCh      = cfg.dataset.sem_fg_stCH
        self.transform    = BaseTransform(cfg, resize_gt=True)
        self.net          = DVIS(cfg)

        if weights is None:
            weights = './vision/instSeg/dvis_'+config.split('_')[1]+'_mc.pth'
            if not os.path.exists(weights):
                print('Please get the weights file ready to use the model')
        self.net.load_weights(weights)
        self.net.eval()

        self.cuda    = torch.cuda.is_available()
        if self.cuda:
            self.net = self.net.cuda()

    def transform_input(self, bgrI, depthI=None):
        '''
        @Func: image transform, mainly normalization and resize if needed.
        '''
        height, width = bgrI.shape[:2]

        # construct virtual mask and target to match transform API
        num_crowds = 0
        masks      = np.zeros([1, height, width], dtype=np.float)
        target     = np.array([[0,0,1,1,0]])
        if depthI is not None:
            assert(bgrI.shape[0] == depthI.shape[0] and bgrI.shape[1]==depthI.shape[1])
            num_crowds = 1
            masks      = np.concatenate([masks, depthI[None, :, :]], axis=0)
            target     = np.concatenate([target, np.asarray([[0,0,1,1,-1]])], axis=0)

        # transform
        img, masks, boxes, labels = self.transform(bgrI, masks, target[:, :4],
                                               {'num_crowds': num_crowds, 'labels': target[:, 4]})

        # concate depthI (if have) to bgrI as network input
        if depthI is not None:
            depthI = np.transpose(masks[-1:], [1,2,0]).astype(np.float32)
            depthI = (depthI-153.)/64.
            img    = np.concatenate([img, depthI], axis=-1)
        return img

    @torch.no_grad()
    def step(self, bgrI, depthI=None):
        '''
        @Param: bgrI -- [ht, wd, 3] in 'BGR' color space
                depthI -- [ht, wd] with value 0 - 255
        '''
        height, width = bgrI.shape[:2]
        depthI = ((depthI*255.)/(depthI.max() + 0.01)).astype(np.uint8)

        normI = self.transform_input(bgrI, depthI) # [ht, wd, ch]
        batch = torch.from_numpy(normI[None, ...]).permute(0, 3, 1, 2) # [1, ch, ht, wd]
        if self.cuda:
            batch = batch.cuda()
        with torch.no_grad():
            preds = self.net(batch)

        cls_logits, mask_logits = preds['cls_logits'][0], preds['proto'][0]
        cls_logits  = cls_logits.view(mask_logits.size(1), -1)
        cls_score   = torch.nn.Softmax(dim=1)(cls_logits) # [BG_cls+N, FG_cls+1]

        preds_score = torch.nn.Softmax(dim=1)(mask_logits)
        preds_score = F.interpolate(preds_score,
                                    size=[height, width],
                                    mode='bilinear',
                                    align_corners=True)
        preds_score = preds_score[0] # [BG_cls+N, ht, wd]

        # remove redundent channel
        _, cls_ids = cls_score.max(axis=1)
        obj_idxes  = cls_ids[self.fg_stCh:].nonzero()
        bg_probs   = preds_score[:self.fg_stCh, :, :]

        if len(obj_idxes) > 0:
            fg_probs   = preds_score[self.fg_stCh:, :, :][obj_idxes, :, :][:, 0, :, :] #[n, ht, wd]
            out_probs  = torch.cat([bg_probs, fg_probs], axis=0) #[BG_cls+n, ht, wd]
            out_scores = cls_score[self.fg_stCh:, :][obj_idxes, :][:, 0, : ]  #[n, k]
        else:
            out_probs, out_scores = bg_probs, cls_score[:0, :]

        # convert to numpy
        if self.cuda:
            net_mask   = preds_score.cpu().detach().numpy().argmax(axis=0)
            out_probs  = out_probs.cpu().detach().numpy()
            out_scores = out_scores.cpu().detach().numpy()
        else:
            net_mask   = preds_score.detach().numpy().argmax(axis=0)
            out_probs, out_scores = out_probs.detach().numpy(), out_scores.detach().numpy()

        # if 'mcsvideo3_voe' == self.dataset_name:
        #     out_probs, out_scores = voe_connect_comp_analysis(out_probs, out_scores,
        #                                                 fg_stCH=1, wronglabel=[6], size_thr=10)
        return {'mask_prob': out_probs,
                'obj_class_score': out_scores,
                'fg_stCh': self.fg_stCh,
                'net-mask': net_mask}


#################################################################
'''
       ******  Demo scripts in below  ******
'''
#################################################################
def display_segment_result(bgrI, depthI, net_out):
    from matplotlib import pyplot as plt

    print('-- object class score: \n', np.round(net_out['obj_class_score'], 3))


    from vision.instSeg.data.config_mcsVideo3_inter import MCSVIDEO_INTER_CLASSES_FG

    SOCCER_INDEX = MCSVIDEO_INTER_CLASSES_FG.index('trophy') + 1
    BOX_INDEX = MCSVIDEO_INTER_CLASSES_FG.index('box') + 1

    cls = np.argmax(net_out['obj_class_score'], axis=1)
    print("all object scores : ", net_out["obj_class_score"])
    print("Soccer Ball idx ", SOCCER_INDEX)
    print("Container idx ", BOX_INDEX)
    print("cls ", cls)
    print("net-mask ", len(net_out['net-mask']))
    print("mask_prob ", len(net_out['mask_prob']))
    n_th_obj = None
    if SOCCER_INDEX in cls:
        n_th_obj = np.where(cls == SOCCER_INDEX)[0]
        print("find trophy in {}th fg object".format(n_th_obj))

    # self.debug_out(bgrI, depthI, ret)


    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(bgrI[..., [2,1,0]])
    ax[0,1].imshow(depthI, cmap='gray')
    ax[1,0].imshow(net_out['net-mask'])
    ax[1,1].imshow(net_out['mask_prob'].argmax(axis=0))

    ax[0,0].set_title('RGB image')
    ax[0,1].set_title('depth image')
    ax[1,0].set_title('net predict mask')
    ax[1,1].set_title('final mask (with cls-score)')
    plt.show()

def demo_interact_segmentation():
    import glob
    import cv2
    import scipy.misc as smisc  # scipy in version <= 1.2.0

    model = MaskAndClassPredictor(dataset='mcsvideo3_inter',
                                  config='plus_resnet50_config_depth_MC',
                                #   weights='./visionmodule/Result/mcsvideo3-inter/weights/test_first_v1/dvis_plus_resnet50_mul_channels_0_26_interrupt.pth')
                                #   weights='./training_logs/dvis_plus_resnet50_mul_channels_26_19600.pth')
                                  weights='./vision/instSeg/dvis_resnet50_mc2.pth')

    img_list = glob.glob('./vision/instSeg/demo/interact/*.jpg')
    for rgb_file in img_list:
        depth_file = rgb_file.replace('original', 'depth')[:-4] + '.png'

        bgrI   = cv2.imread(rgb_file)
        depthI = smisc.imread(depth_file, mode='P')
        ret    = model.step(bgrI, depthI)

        display_segment_result(bgrI, depthI, ret)

def demo_voe_segmentation():
    import glob
    import cv2
    import scipy.misc as smisc  # scipy in version <= 1.2.0

    model = MaskAndClassPredictor(dataset='mcsvideo3_voe',
                                  config='plus_resnet50_config_depth_MC',
                                  weights='./vision/instSeg/dvis_resnet50_mc_voe.pth')

    img_list = glob.glob('./vision/instSeg/demo/voe/*.jpg')
    for rgb_file in img_list:
        depth_file = rgb_file.replace('original', 'depth')[:-4] + '.png'

        bgrI   = cv2.imread(rgb_file)
        depthI = smisc.imread(depth_file, mode='P')
        ret    = model.step(bgrI, depthI)

        display_segment_result(bgrI, depthI, ret)


if __name__=='__main__':

    # demo_voe_segmentation()

    demo_interact_segmentation()

