import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .iou_loss import IoULoss
from .instance_loss import BinaryLoss, PermuInvLoss
from .regularize_loss import MumfordShahLoss
from .classify_mc_loss import ClassifyMCLoss
from .evaluate import Evaluate

class LossEvaluate(nn.Module):
    '''
    compute loss:
        1) binary loss to separate BG and FG
        2) permutation invariant loss to separate different instances and group pixels in one obj.
        3) M-S loss to regularize the segmentation level
        4) use iou-loss to force predicted value close to 1 on corresponding GT.
    '''
    def __init__(self, config, class_weights=None, ignore_label=[255], fg_stCH=1):
        """
        @Param: config -- loss related configurations
                class_weights -- dict with key is the class label and value is the weight
                ignore_label -- ignore class label
                fg_stCH -- it could > 1 for panoptic segmentation
        """

        super(LossEvaluate, self).__init__()
        self.config    = config
        self.fg_stCH   = fg_stCH
        self.softmax2d = nn.Softmax2d()

        self.class_weights = [class_weights[ele] for ele in sorted(class_weights.keys()) \
                                                             if ele not in ignore_label]
        self.class_weights = torch.FloatTensor(self.class_weights).cuda()

        # loss functions
        self.Binary_loss, self.PI_loss, self.Cls_loss = None, None, None
        self.MS_loss, self.IoU_loss, self.Evaluate = None, None, None
        self.setupMultiChannelLosses(self.class_weights, fg_stCH=fg_stCH)

    def setupMultiChannelLosses(self, class_weights, fg_stCH=1):
        if 'pi_alpha' in self.config and self.config['pi_alpha']>0:
            self.PI_loss = PermuInvLoss(class_weights=class_weights,
                                        margin=self.config['pi_margin'],
                                        pi_pairs=self.config['pi_smpl_pairs'],
                                        smpl_wght_en=self.config['pi_smpl_wght_en'],
                                        pos_wght=self.config['pi_pos_wght'],
                                        loss_type=self.config['pi_loss_type'],
                                        FG_stCH=fg_stCH)

        if 'binary_alpha' in self.config and self.config['binary_alpha']>0:
            if self.config['binary_loss_type'] == 'CE':
                class_weights_binary = torch.ones_like(class_weights[:fg_stCH+1])
                class_weights_binary[-1] = class_weights[fg_stCH:].max()
            else:
                class_weights_binary = None

            self.Binary_loss = BinaryLoss(margin=self.config['binary_margin'],
                                        FG_stCH=fg_stCH,
                                        loss_type=self.config['binary_loss_type'],
                                        weights=class_weights_binary)

        if 'regul_alpha' in self.config and self.config['regul_alpha']>0:
            self.MS_loss = MumfordShahLoss()

        if 'cls_en' in self.config and self.config['cls_en']==1:
            self.Cls_loss = ClassifyMCLoss(class_weights=class_weights, fg_stCH=fg_stCH,
                                            cls_pos_iou_thr=self.config['cls_pos_iou_thr'])

        if 'iou_alpha' in self.config and self.config['iou_alpha']>0:
            self.IoU_loss = IoULoss(FG_stCH=fg_stCH, compute_loss=True)

        if self.Cls_loss is not None and 'eval_en' in self.config and self.config['eval_en']:
            self.Evaluate = Evaluate(size_thrs=self.config['eval_size_thrs'],
                                     iou_thr  =self.config['eval_iou_thr'],
                                     eval_classes=self.config['eval_classes'])
        return


    def stableSoftmax(self, logits):
        max_logits = logits.max(dim=1, keepdim=True)[0]
        max_logits.require_grad = False
        return self.softmax2d(logits - max_logits)

    @torch.no_grad()
    def create_global_slope_plane(self, ht, wd):
        '''
        return a tensor in shape [1, 1, ht, wd] with value = row+col
        '''
        slopeX = np.cumsum(np.ones([ht, wd]), axis=1)
        slopeY = np.cumsum(np.ones([ht, wd]), axis=0)
        plane  = slopeX + slopeY
        return torch.FloatTensor(plane[np.newaxis, np.newaxis, ...])

    @torch.no_grad()
    def mapLocalMask2Global(self, batch_size, loc_mask_logits, loc_cls_logits,
                                    loc_rois, fht, fwd):
        '''
        @Func:
        @Param: batch_size -- batch size
                loc_mask_logits -- tensor in size [N, 1, roi_ht, roi_wd]
                loc_cls_logits -- tensor in size [N, num_classes]
                loc_rois -- tensor in size [N, 5], as (bk, x0,y0,x1,y1)
                fht / fwd -- height / width of the full image
        @Output: preds_mask -- [N, 1, ht, wd]
                 preds_cls --
        '''
        # local to global
        preds_mask, preds_cls, obj_cnts = [None]*batch_size, [None]*batch_size, [0]*batch_size
        N = loc_mask_logits.size(0)
        for k in range(N):
            bk, x0, y0, x1, y1 = loc_rois[k].int()
            nht, nwd = y1-y0+1, x1-x0+1
            tmp = torch.zeros([1, 1, fht, fwd])
            tmp[:,:,y0:y1+1, x0:x1+1] = F.interpolate(nn.Sigmoid()(loc_mask_logits[k:k+1]),
                                                      size=[nht, nwd],
                                                      mode='bilinear',
                                                      align_corners=True)
            obj_cnts[bk] += 1
            if preds_mask[bk] is None:
                preds_mask[bk] = [tmp>0.5]
                preds_cls[bk]  = [loc_cls_logits[k][None, None, :]]
            else:
                preds_mask[bk].append(tmp>0.5)
                preds_cls[bk].append(loc_cls_logits[k][None, None, :])

        # batch stack
        max_cnt = max(obj_cnts)
        for k in range(batch_size):
            if obj_cnts[k] < max_cnt:
                comp_mask = torch.zeros([1, max_cnt-obj_cnts[k], fht, fwd], dtype=torch.bool)
                preds_mask[k].append(comp_mask)
                comp_cls = torch.zeros([1, max_cnt-obj_cnts[k], loc_cls_logits.size(1)])
                preds_cls[k].append(comp_cls)
            preds_mask[k] = torch.cat(preds_mask[k], dim=1)
            preds_cls[k] = torch.cat(preds_cls[k], dim=1)
        preds_mask = torch.cat(preds_mask, dim=0)
        preds_cls = torch.cat(preds_cls, dim=0)

        return {'mask': preds_mask,
                'cls': preds_cls}


    def forward(self, preds, targets, preds_cls=None, target_boxes=None):
        ''' Compute loss to train the network and report the evaluation metric
        Params: preds -- list of instance label prediction in different FPN scale.
                targets -- tensor in [bs, ch, ht, wd] with full GT objects.
                preds_cls -- list of classify dict prediction from FPN
                target_boxes -- list of GT object bboxes [ch, 5],  as [x0,y0,x1,y1,cls_id-1]
        '''
        # prepare target in original size
        target_ids = torch.stack(target_boxes, axis=0)[:, :, -1] #[bs, ch]
        target_labelI = targets.max(axis=1, keepdim=True)[1] #[bs, 1, ht, wd]

        # compute loss
        if len(preds) > 1:
            max_num_objs  = max(self.fg_stCH, target_labelI.max())
            targets_BG    = targets[:, :self.fg_stCH,:,:]
            target_ids_BG = target_ids[:, :self.fg_stCH].int()

            targets_FG    = targets[:, self.fg_stCH:(max_num_objs+1),:,:]
            target_ids_FG = target_ids[:, self.fg_stCH:(max_num_objs+1)].int() #[bs, N]

            bs, ch, _, _    = targets_FG.size()
            target_areas_FG = targets_FG.view(bs, ch, -1).sum(axis=-1) #[bs, N]
            existI_FG       = targets_FG.sum(axis=1) #[bs, ht, wd]

            ret = dict()
            for k in range(len(preds)):
                scales = self.config['fpn_scales'][k]
                hit_target = self.target_hit_scale(targets_FG, target_ids_FG, target_areas_FG,
                                                    lower=scales[0], upper=scales[1])
                FG_holeI = existI_FG - hit_target['targets'].sum(axis=1)
                new_target = torch.cat([targets_BG, hit_target['targets']], axis=1)
                new_target[:, 0] += FG_holeI
                new_ids    = torch.cat([target_ids_BG, hit_target['classes']], axis=1)

                tmp_cls = preds_cls[k] if preds_cls is not None else None
                if k == 0:
                    tmp_ret, weightI = self.process_onescale(preds[k], new_target, tmp_cls, new_ids)
                else:
                    tmp_ret, _ = self.process_onescale(preds[k], new_target, tmp_cls, new_ids)

                for key in tmp_ret:
                    ret[key] = tmp_ret[key] if key not in ret else  (ret[key] + tmp_ret[key])
        else:
            tmp_cls = preds_cls[0] if preds_cls is not None else None
            ret, weightI = self.process_onescale(preds[0], targets, tmp_cls, target_ids)


        # tfboard visual tensors
        ret['preds_0'] = preds[len(preds)//2]
        ret['preds'] = [ele.max(axis=1, keepdim=True)[1] for ele in preds]
        ret['gts']= target_labelI
        ret['wghts'] = weightI

        return ret

    def target_hit_scale(self, targets, target_ids, target_areas, lower=0, upper=2048):
        ''' select out target objects with size in given bounds area
        '''
        bs, _, ht, wd = targets.size()
        keep = (target_areas > lower) * (target_areas <= upper) # [bs, K]

        keep_cnt = keep.sum(axis=1) # [bs]
        max_cnt = keep_cnt.max()
        hit_targets, hit_ids = [], []
        if max_cnt == 0:
            max_cnt = 1

        for bk in range(targets.size(0)):
            new_targets, new_ids = [], []
            for k, ele in enumerate(keep[bk]):
                if ele:
                    new_targets.append(targets[bk, k:k+1])
                    new_ids.append(target_ids[bk, k:k+1])
            if keep_cnt[bk] < max_cnt:
                complement_0 = torch.zeros([max_cnt-keep_cnt[bk], ht, wd])
                complement_1 = torch.zeros([max_cnt-keep_cnt[bk]], dtype=torch.int32)
                new_targets.append(complement_0)
                new_ids.append(complement_1)

            hit_targets.append(torch.cat(new_targets, axis=0))
            hit_ids.append(torch.cat(new_ids, axis=0))

        hit_targets = torch.stack(hit_targets, axis=0)
        hit_ids = torch.stack(hit_ids, axis=0)
        return {'targets':hit_targets,
                'classes':hit_ids}

    def process_onescale(self, preds, targets, pred_cls=None, target_ids=None):
        ''' Compute loss to train the network and report the evaluation metric
        Params: preds -- tensor for instance prediction, [bs, K, ht, wd].
                targets -- tensor in [bs, ch, h', w'] with full GT objects.
                pred_cls -- dict, classify prediction
                target_ids -- tensor in [bs, ch], GT categoryID for each target
        '''
        #def _compute_weights_one_instance(mask, sem_id, base_cnt):
        #    inst_wght = np.cbrt(base_cnt/(mask.sum()+1.))
        #    sem_wght  = 1.0 if sem_weights is None else sem_weights[sem_id]
        #    return np.clip(sem_wght*inst_wght, 1.0, 10.0)

        bs, _, ht, wd      = preds.size()
        targets_rs  = self.resize_GT(targets, ht, wd) #[bs, ch, ht, wd]
        cnts        = targets_rs.sum(axis=-1).sum(axis=-1) #[bs, ch]
        inst_wght   = torch.pow((ht*wd)/(cnts+1), 1/3.)
        sem_wght    = self.class_weights[target_ids.view(-1).long()].reshape(cnts.size())
        weights     = torch.clamp(inst_wght*sem_wght, 1.0, 30.0)[:, :, None, None]# [bs, ch, 1, 1]
        weightI     = (weights*targets_rs).sum(axis=1, keepdim=True)

        ret = self.process_step_multiChannel(preds, targets_rs, weightI, pred_cls, target_ids)
        return ret, weightI

    @torch.no_grad()
    def resize_GT(self, targets, nht, nwd):
        """
        @Func: resize GT make sure BG channels has no overlap 1.
             if bg_ch == 1, perform bilinear intepolation on each channel, and >0.5 to obtain binary mask
             if bg_ch > 1, perform nearest intepolation on BG channels by combining bg channels into one channel
                           perform bilinear intepolation on FG channels.
        """
        if self.fg_stCH == 1:
            gts_rs  = F.interpolate(targets, size=[nht, nwd], mode='bilinear', align_corners=True)
        else:
            _, targets_bg = targets[:, :-1, :, :].max(axis=1, keepdim=True)
            targets_bg[targets_bg >=self.fg_stCH] = self.fg_stCH
            gts_rs_bg  = F.interpolate(targets_bg.float(), size=[nht, nwd], mode='nearest') # [bs,ch, ht, wd]
            gts_rs_bg  = torch.eye(self.fg_stCH+1)[gts_rs_bg[:,0,:,:].long(), :] # [bs, ht, wd, ch]
            gts_rs_bg = gts_rs_bg.permute([0,3,1,2])[:, :self.fg_stCH, :, :] #[bs, ch, ht, wd]

            targets_fg = targets[:, self.fg_stCH:, :, :]
            gts_rs_fg  = F.interpolate(targets_fg,
                                       size=[nht, nwd],
                                       mode='bilinear', align_corners=True)
            gts_rs = torch.cat([gts_rs_bg, gts_rs_fg], axis=1)

        return gts_rs

    def process_step_multiChannel(self, preds, targets, weights, preds_cls, target_ids,
                                        is_training=True):
        ''' Compute loss to train the network and report the evaluation metric
        Params: preds -- tensor for instance prediction, [bs, K, ht, wd].
                targets -- tensor in [bs, ch, ht, wd] with full GT objects.
                weights -- tensor in [bs, 1, ht, wd]
                preds_cls -- dict, if 'noROI' mode, dict from refineNet_noROI output
                                   if 'ROI' mode, dict from refineNet output
                target_ids -- tensor in [bs, ch], GT categoryID for each target
                is_training -- if False, only run Evaluate
        '''
        bs, ch, ht, wd = preds.size()
        gts_onehot = (targets>0.5).int() # larger objects have smaller channel ID

        # compute loss 1
        ret = {}
        if is_training:
            if self.config['binary_loss_type'] == 'l1':
                preds = self.softmax2d(preds)

            if self.Binary_loss is not None:
                loss = self.Binary_loss(preds, gts_onehot, weights=weights)
                ret['binary'] = loss * self.config['binary_alpha']

            # process on network output after binary loss
            if self.config['binary_loss_type'] != 'l1':
                preds = self.softmax2d(preds)

            if self.PI_loss is not None:
                pi_weights= None if self.config['pi_smpl_wght_en']>0 else weights
                loss = self.PI_loss(preds, gts_onehot,
                                    target_ids=target_ids,
                                    weights=pi_weights,
                                    BG=self.config['pi_hasBG'])
                if loss is not None:
                    ret['pi'] = loss['loss'] * self.config['pi_alpha']
                    ret['eval_pi0'] = loss['eval_pi0']
                    ret['eval_pi1'] = loss['eval_pi1']

            if self.MS_loss is not None:
                loss = self.MS_loss(preds)
                if loss is not None:
                    ret['regul'] = loss * self.config['regul_alpha']

            preds_rmEdge   = preds*((weights>0).float()) # remove edge garbage from batch resize
            if self.IoU_loss is not None:
                loss = self.IoU_loss(preds_rmEdge, gts_onehot, return_hungarian_map=False)
                iou_iou, iou_indices =loss['iou-iou'], loss['indices']
                if loss['loss'] is not None:
                    ret['iou'] = loss['loss'] * self.config['iou_alpha']

            if self.Cls_loss is not None and preds_cls is not None and 'iou' in ret:
                loss = self.Cls_loss(preds_cls, target_ids, iou_indices, iou_iou, preds)
                ret['cls_cls'] = loss['cls_loss']*self.config['cls_cls_alpha']
                ret['cls_iou'] = loss['iou_loss']*self.config['cls_iou_alpha']
        else:
            preds_rmEdge = self.softmax2d(preds)*((weights>0).float())

        # evaluation
        if self.Evaluate is not None and preds_cls is not None:
            target_cls_ids = target_ids - self.fg_stCH + 1
            target_cls_ids[target_cls_ids< 0] = 0
            evalV = self.Evaluate(preds_rmEdge[:, self.fg_stCH:, :, :],
                                  gts_onehot[:, self.fg_stCH:, :, :],
                                  preds_cls['cls_logits'][:, self.fg_stCH:, :],
                                  target_cls_ids[:, self.fg_stCH:])

            ret['eval_prec'] = evalV['prec']
            ret['eval_rec'] = evalV['rec']
            ret['eval_acc'] = evalV['acc']

        return ret

