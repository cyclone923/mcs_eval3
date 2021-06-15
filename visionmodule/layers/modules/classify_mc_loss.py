
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .classify_loss import CrossEntropyLoss
from torchvision.ops import roi_align


def adjust_smooth_l1_loss(y_pred, theta=0.1):
    # small gradient when close to 0, constant gradient in large value zone
    less_grad_factor = 1./(2*theta)
    less_loss_bias   = less_grad_factor * theta**2
    less_than_theta  = (y_pred < theta).float()
    loss = (less_than_theta*(y_pred**2)*less_grad_factor) + \
           (1-less_than_theta)*(y_pred-theta + less_loss_bias)
    return loss

class ClassifyMCLoss(nn.Module):
    ''' this class compute loss for the classify branch in multi-channel architecture, which includes
     CE_cls_loss for classification, and l1_loss for iou prediction.
    '''
    def __init__(self, class_weights=None, fg_stCH=1, cls_pos_iou_thr=0.2):
        super(ClassifyMCLoss, self).__init__()
        self.fg_stCH         = fg_stCH
        self.cls_pos_iou_thr = cls_pos_iou_thr

        self.class_weights_cls = None
        if class_weights is not None:
            self.class_weights_cls = torch.ones_like(class_weights[fg_stCH-1:])
            self.class_weights_cls[1:] = class_weights[fg_stCH:]
        self.CE_loss_cls = CrossEntropyLoss(weight=self.class_weights_cls, reduction='none')


    def forward(self, preds, target_ids, map_indices, map_ious, pred_mask_prob):
        '''
        @Param:
            preds -- dict includes: 'cls_logits' -- [bs, ch, num_classes]
                                    'iou_scores' -- [bs, ch, 1]
            target_ids -- [bs, ch']
            map_indices -- tensor in size [bs, 2, ch], with [0] for pred, [1] for GT
            map_ious -- [bs, ch]
            pred_mask_prob -- [bs, ch, ht, wd]
        '''
        if self.class_weights_cls is None:
            self.class_weights_cls = [1.0] * preds['cls_logits'].size(-1)

        map_dict = self.construct_classify_GT(preds, target_ids,
                                              map_indices, map_ious, pred_mask_prob)
        iou_diff = torch.abs(map_dict['preds_iou'] - map_dict['gts_iou'])
        iou_loss = adjust_smooth_l1_loss(iou_diff) #[N, 1]
        cls_loss = self.CE_loss_cls(map_dict['preds_cls'], map_dict['gts_cls'].long()) #[N, 1]

        # weights
        wght_sum = map_dict['weights'].sum()+1e-4  # divide 0 protect
        iou_loss = (iou_loss*map_dict['weights']).sum()/wght_sum
        cls_loss = (cls_loss*map_dict['weights']).sum()/wght_sum

        return {'iou_loss': iou_loss, 'cls_loss': cls_loss}


    def construct_classify_GT(self, preds, target_ids,
                                    map_indices, map_iou,
                                    pred_mask_probs, entity_prob_thr=0.1, remove_thr=0.9):
        '''
        @Func:  extract prediction and its corresponding targets for compute classify loss
                here class_id recount from 1 for all FG categories.
        @Param:
                preds -- dict includes: 'cls_logits' -- [bs, ch, num_classes]
                                        'iou_scores' -- [bs, ch, 1]
                target_ids -- tensor in size [bs, ch'], original input target classIds
                map_indices -- tensor in size [bs, 2, ch], with [0] for pred, [1] for GT
                map_iou -- tensor in size [bs, ch]
                pred_mask_probs -- tensor in size [bs, ch, ht, wd],
        @Output:
                a dict includes:
                    weights
                    preds_iou -- tensor in size [bs*ch, num_classes]
                    gts_iou --  gt iou for train classify net
                    preds_cls -- tensor in size [bs*ch, num_classes]
                    gts_cls --  class_id for train classify net. ref IoU
        '''
        pred_cls_logits = preds['cls_logits']
        pred_iou_logits = preds['iou_scores']

        bs, ch, num_classes = pred_cls_logits.size()
        pred_mask_maxProb   = torch.max(pred_mask_probs.view(bs, ch, -1), axis=-1)[0] #[bs, ch]

        weights = []
        preds_iou, gts_iou, preds_cls, gts_cls  = [], [], [], []
        for b in range(bs):
            for k in range(self.fg_stCH, ch):
                pj, gj = map_indices[b][0][k], map_indices[b][1][k]

                cls = max(target_ids[b][gj]*0, target_ids[b][gj] - self.fg_stCH + 1)
                iou = map_iou[b][k]

                # remove empty prediction with a ratio 'remove_thr', so that focus on training
                if pred_mask_maxProb[b, pj] < entity_prob_thr and torch.rand(1) < remove_thr:
                    wght = self.class_weights_cls[cls.long()] * 0
                elif iou < self.cls_pos_iou_thr:
                    wght = self.class_weights_cls[(cls*0).long()]
                else:
                    wght = self.class_weights_cls[cls.long()] + self.class_weights_cls[0]

                preds_iou.append(pred_iou_logits[b, pj])
                gts_iou.append(iou)
                preds_cls.append(pred_cls_logits[b, pj])
                gts_cls.append(cls)
                weights.append(wght)

        return {'preds_cls': torch.stack(preds_cls),
                'gts_cls':   torch.stack(gts_cls),
                'preds_iou': torch.stack(preds_iou),
                'gts_iou':   torch.stack(gts_iou),
                'weights':   torch.stack(weights)   }

