import os
import os.path as osp
import sys
import cv2
import random
import numpy as np
from scipy import misc as smisc
from glob import glob

import torch
import torch.utils.data as data
import torch.nn.functional as F

from vision.instSeg.base_dataset import Detection
from vision.instSeg.base_dataset import FromImageAnnotationTransform as AnnotationTransform

class MCSVIDEODetection(Detection):
    """`
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (semImg, instImg) and transforms it to bbox+cls.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, mask_out_ch=1, info_file=None, option=None,
                 transform=None, target_transform=None,
                 dataset_name='dummy', running_mode='test', model_mode='InstSeg'):
        '''
        Args:running_mode: 'train' | 'val' | 'test'
             model_mode: 'InstSeg' | 'SemSeg' | 'ObjDet'
        '''
        super(MCSVIDEODetection, self).__init__(image_path,
                                            mask_out_ch,
                                            option.sem_weights,
                                            transform,
                                            AnnotationTransform(option),
                                            running_mode,
                                            model_mode,
                                            option.sem_fg_stCH)
        self.ignore_label   = option.ignore_label
        self.name           = dataset_name
        self.image_set      = running_mode
        self.ids            = self._load_image_set_index(info_file)
        self.read_depth     = option.extra_input

    def _load_image_set_index(self, info_file):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        base_dir = self.root
        with open(info_file) as f:
            fdir_list = [x.strip() for x in f.readlines()]

        # get image list from each sub-folder.
        key = '.jpg'
        image_set_index = []
        for fdir in fdir_list:
            glob_imgs = glob(osp.join(base_dir, fdir, '*'+key))
            img_list = [osp.join(fdir, osp.basename(v).split(key)[0]) for v in glob_imgs]
            image_set_index += img_list

        return image_set_index

    def _readLabelImage(self, fname):
        '''
        @func: given fname, read the color maskImage and parse to instanceI and semanticI
        '''
        fdir, ori_name = os.path.dirname(fname), os.path.basename(fname)
        sem_file  = osp.join(self.root, fdir, ori_name.replace('original', 'cls')+'.png')
        semI = smisc.imread(sem_file, mode='P')

        inst_file = osp.join(self.root, fdir, ori_name.replace('original', 'inst')+'.png')
        instI = smisc.imread(inst_file, mode='P')

        return semI, instI

    def save_subpath(self, index, result_path='', subPath=''):
        fname = self.ids[index]
        sub_folder, fname = os.path.dirname(fname), os.path.basename(fname)
        result_path =  osp.join(result_path, subPath, sub_folder)
        os.makedirs(osp.join(result_path), exist_ok=True)
        return {'fname': fname.replace('original', ''),
                'out_dir': result_path,
                'file_key': self.ids[index] }

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        fpath = self.ids[index]
        sub_folder, fname = os.path.dirname(fpath), os.path.basename(fpath)

        # read image
        img = cv2.imread(osp.join(self.root, sub_folder, fname + '.jpg'))
        height, width, _ = img.shape

        if self.read_depth:
            depth_file = osp.join(self.root, sub_folder, fname.replace('original', 'depth') + '.png')
            if os.path.exists(depth_file):
                depthI = smisc.imread(depth_file, mode='P')
                depthI = depthI[:, :, None]
            else:
                sys.exit("can not find the depth file", depth_file)

        num_crowds = 0
        if self.has_gt:
            anns = self.pull_anno(index)
            semI, masks, target = anns['sem'], anns['inst_mask'], anns['bbox']
        if not self.has_gt or target is None:
            semI   = None
            masks  = np.zeros([1, height, width], dtype=np.float)
            target = np.array([[0,0,1,1,0]])

        if self.read_depth:
            num_crowds = 1
            masks = np.concatenate([masks, np.transpose(depthI, [2,0,1])], axis=0)
            target = np.concatenate([target, np.asarray([[0,0,1,1,-1]])], axis=0)

        # add BG semantic channels, for panoptic segmentation
        if semI is not None:
            sem_bgs =np.asarray([[0,0,1,1,0]]*self.sem_fg_stCH)
            sem_bg_maskI = np.zeros([self.sem_fg_stCH, height, width])
            for k in range(self.sem_fg_stCH):
                sem_bg_maskI[k] = (semI==k).astype(np.float)
            masks  = np.concatenate([sem_bg_maskI, masks], axis=0)
            target = np.concatenate([sem_bgs, target], axis=0)

        if self.transform is not None:
            img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                                 {'num_crowds': num_crowds, 'labels': target[:, 4]})
            # num_crowds is stored inheirted from coco dataset
            num_crowds = labels['num_crowds']
            labels     = labels['labels']
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        # separate depthI from masks, and concatenate it to image.
        if self.read_depth:
            depthI = np.transpose(masks[-1:], [1,2,0]).astype(np.float32)
            depthI = (depthI-153.)/64.
            img    = np.concatenate([img, depthI], axis=-1)
            num_crowds, masks, target = 0, masks[:-1], target[:-1]

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            dict of network input
        '''
        fpath = self.ids[index]
        sub_folder, fname = os.path.dirname(fpath), os.path.basename(fpath)
        bgrI = cv2.imread(osp.join(self.root, sub_folder, fname + '.jpg'))

        if self.read_depth:
            depth_file = osp.join(self.root, sub_folder, fname.replace('original', 'depth') + '.png')
            depthI = smisc.imread(depth_file, mode='P')[..., None]
            depthI = np.concatenate([depthI, depthI, depthI], axis=-1)

            return {'rgb': bgrI[:, :, [2,1,0]], 'depth': depthI}
        else:
            return {'rgb': bgrI[:, :, [2,1,0]]}

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            dict of annotations -- bbox: list of bbox in [x0,y0,x1,y1,clsId]
                                   inst_mask: object mask in array [N, ht, wd]
                                   sem: sem label image in [ht, wd]
                                   inst: inst label image in [ht, wd]
        '''
        fpath = self.ids[index]
        semI, instI = self._readLabelImage(self.ids[index])
        height, width = semI.shape[:2]

        # obtain bbox and mask
        if self.target_transform is not None:
            trans_src = [semI, instI]
            target = self.target_transform(trans_src, width, height)
            target = np.array(target)
            if len(target) == 0:
                target, masks = None, None
            else:
                # instance binary masks in different channels
                cor_instI = trans_src[1]
                masks = np.eye(cor_instI.max()+1)[cor_instI]
                eff_chs = [0] + [ele for ele in np.unique(cor_instI) if ele > 0]
                masks = masks[..., eff_chs]
                masks = np.transpose(masks[:,:,1:], [2,0,1]).astype(np.float)
        else:
            target, masks = None, None

        return {'bbox': target,
                'inst_mask': masks,
                'sem': semI,
                'inst': instI}

