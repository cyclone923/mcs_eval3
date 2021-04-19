import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .classify_loss import CrossEntropyLoss
from .python_func import create_pairwise_conv_kernel

def adjust_smooth_l1_loss(y_pred, theta=0.1):
    # small gradient when close to 0, constant gradient in large value zone
    less_grad_factor = 1./(2*theta)
    less_loss_bias   = less_grad_factor * theta**2
    less_than_theta  = (y_pred < theta).float()
    loss = (less_than_theta*(y_pred**2)*less_grad_factor) + \
           (1-less_than_theta)*(y_pred-theta + less_loss_bias)
    return loss


def pi_l1_loss(pi_pred, pi_target, margin=1.0, pos_wght=3.0):
    '''
    Params: pi_pred -- float tensor in size [N, N]
            pi_target -- float tensor in size [N, N], with value {0, 1}, which
                        ** 1 for pixels belonging to different object
                        ** 0 for pixels belonging to same object
            margin/pos_wght -- float
    '''
    loss_pi_1 = adjust_smooth_l1_loss(F.relu(margin-pi_pred))
    loss_pi_0 = adjust_smooth_l1_loss(pi_pred)

    loss = loss_pi_1*pi_target*pos_wght + loss_pi_0*(1.0-pi_target)
    return loss

def pi_exp_loss(pi_pred, pi_target, margin=1.0, pos_wght=3.0):
    '''
    @ refering paper 'semantic instance segmentation via deep matric learning
    Params: pi_pred -- float tensor in size [N, N]
            pi_target -- float tensor in size [N, N], with value {0, 1}
                        ** 1 for pixels belonging to different object
                        ** 0 for pixels belonging to same object
            margin/pos_wght -- float
    '''
    pi_diff = pi_pred
    pi_sims = 2.0*(margin-torch.sigmoid(pi_diff))

    loss = -(pi_target       * torch.log(1.0-pi_sims)*pos_wght + \
             (1.0-pi_target) * torch.log(pi_sims))
    return loss


class BinaryLoss(nn.Module):
    '''This class computes the Binary loss to force BG pixels close to 0 and FG pixels far away.
    '''
    def __init__(self, margin=2.0, FG_stCH=1, loss_type='l1', weights=None):
        """
        margin: minimum distance between FG/BG if prediction with 1 channel
        FG_stCH: start channel of FG objects on prediction with multiple channels
        loss_type: 'l1' | 'CE', works for prediction with multiple channel.
                   if 'l1', prediction is expected to be softmax2d output.
                   if 'CE', prediction is expected to be net logits
                   if prediction has channel=1,
        weights: if not None, a tensor of size C
        """
        super(BinaryLoss, self).__init__()
        self.margin = margin
        self.FG_stCH = FG_stCH
        self.loss_type = loss_type if FG_stCH>1 else 'l1'
        if self.loss_type == 'CE':
            self.CE_loss = CrossEntropyLoss(weight=weights, reduction='none')

    def compute_loss(self, preds, targets):
        '''
        Params:
            preds --  tensor, [bs, ch, ht, wd], ch> self.FG_stCH
            targets -- tensor, [bs, ch', ht, wd], ch' > self.FG_stCH
        '''
        bs, ch, ht, wd = preds.size()
        # preds is probability
        if self.loss_type == 'l1':
            preds_0   = preds[:, :self.FG_stCH, :, :]
            targets_0 = targets[:, :self.FG_stCH, :, :].float()

            loss      = adjust_smooth_l1_loss(torch.abs(targets_0-preds_0))
            loss      = loss.sum(axis=1, keepdim=True)

        else: # 'softmax cross-entropy'
            preds_0   = preds[:, :self.FG_stCH, :, :].float()
            preds_1,_ = preds[:, self.FG_stCH:, :, :].float().max(axis=1, keepdim=True)

            targets_0 = targets[:, :self.FG_stCH, :, :]
            targets_1 = 1-targets_0.max(axis=1, keepdim=True)[0]
            _, target_id = torch.cat([targets_0, targets_1], axis=1).max(axis=1)

            loss = self.CE_loss(torch.cat((preds_0, preds_1), axis=1), target_id) #[bs, ht, wd]
            loss = loss[:, None, :, :] # [bs, 1, ht, wd]
        return loss

    def forward(self, preds, targets, weights=None):
        '''
        Params:
            preds --  tensor, [bs, ch, ht, wd], ch> self.FG_stCH
            targets -- tensor, [bs, ch', ht, wd], ch' > self.FG_stCH
            weights -- tensor, [bs, 1, ht, wd]
        '''
        bs, ch, ht, wd = preds.size()
        loss = self.compute_loss(preds, targets)

        if weights is not None:
            loss = torch.mul(loss, weights).sum()/(weights.sum()+1e-4)
        else:
            loss = loss.mean()

        return loss.float()


class PermuInvLoss(nn.Module):
    '''This class compute the permutation-invariant loss between the targets and the predictions.
        The pixels used to compute pi_loss is sampled on each objects
        It encourages pixels in same instance close to each other,
                      pixels from different instances far away from each other.

    '''
    def __init__(self, class_weights=None, margin=1.0, pi_pairs=4096, avg_num_obj=8,
                       smpl_wght_en=0.0, pos_wght=3.0, loss_type='l1', FG_stCH=1):
        super(PermuInvLoss, self).__init__()
        self.class_weights = class_weights
        self.margin    = margin
        self.pi_pairs  = pi_pairs
        self.avg_num_obj = avg_num_obj
        self.smpl_wght_en   = smpl_wght_en
        self.pos_wght  = pos_wght
        self.loss_type = loss_type
        self.fg_stCH = FG_stCH

    def sampling_over_objects(self, targets_onehot, target_classes, BG=0):
        '''
        Params:
            targets_onehot -- tensor in [N, ch] with integers values.
            target_classes -- if not None, in size [ch], ch is No. of Objs in targets.
            BG -- if True, BG is counted as one object.
        '''
        eff_idx, smpl_wght = [], []
        cnts    = targets_onehot.sum(axis=0)  #[ch]
        num_obj = (cnts>0).sum() if BG else (cnts[self.fg_stCH:]>0).sum()

        # sample over each object
        avg     = self.pi_pairs//num_obj
        for k in range(cnts.size(0)):
            if cnts[k]==0 or (BG==0 and k<self.fg_stCH):
                continue

            # sample index on current object
            idx = targets_onehot[:, k].nonzero()
            perm = torch.randperm(idx.size(0))
            cur_sel = idx[perm][:avg]
            smpl_size = torch.FloatTensor([cur_sel.size(0)]).cuda()
            obj_wght = torch.pow(self.pi_pairs/(smpl_size + 1.), 1./3)
            if target_classes is not None and self.class_weights is not None:
                obj_wght += self.class_weights[target_classes[k].long()]

            # add into the whole stack.
            eff_idx.append(cur_sel)
            smpl_wght.append(torch.ones(cur_sel.size(0), 1, dtype=torch.float)*obj_wght)

        if len(eff_idx) == 0:
            return None, None
        else:
            eff_idx = torch.cat(eff_idx, axis=0).squeeze() #[N]
            if len(eff_idx.size())==0 or eff_idx.size(0)==1:
                import pdb; pdb.set_trace()
            smpl_wght = torch.cat(smpl_wght, axis=0) #[N,1]
            return eff_idx, smpl_wght

    def forward(self, preds, targets, target_ids=None, weights=None, BG=False, sigma=1e-2):
        ''' Compute the permutation invariant loss on pixel pairs if both pixels are in the instances.
        Params:
            preds -- [bs, ch, ht, wd], it is prob from Softmax()
            targets -- [bs, ch', ht, wd]. onehot matrix
            weights -- [bs, 1, ht, wd]
            target_ids -- [bs, ch']
            BG -- if True, treat BG as one instance.
                | if False, don't sample point on BG pixels,
                            and compute difference only on FG channels if ch>1
        '''
        all_loss = []
        bs, ch, ht, wd = preds.size()

        # reshape
        preds_1D = preds.view(bs, ch, ht*wd).permute(0, 2, 1) # in size [bs, N, ch]
        targets_1D = targets.view(bs, -1, ht*wd).permute(0, 2, 1) # in size [bs, N, ch']
        if weights is not None:
            weight_1D = weights.view(bs, -1, 1)

        # compute loss for each sample
        all_loss = []
        eval_pi0, eval_pi1 = [], []
        for b in range(bs):
            with torch.no_grad():
                smpl_idx, smpl_wght = self.sampling_over_objects(targets_1D[b], target_ids[b], BG=BG)

            if smpl_idx is None:
                continue

            # compute pairwise differences over pred/target/weight
            #'''cosine distance = 1-(x1x2)/(|x1|*|x2|)'''
            smpl_pred = preds_1D[b][smpl_idx]
            pred_numi = torch.matmul(smpl_pred, smpl_pred.permute(1,0)) #[N, N]
            pred_tmp  = smpl_pred.pow(2).sum(axis=1).pow(0.5) # size [N]
            pred_demi = torch.matmul(pred_tmp[:,None], pred_tmp[None,:]) #[N,N]
            pi_pred   = 1 - pred_numi/pred_demi #[N, N]

            with torch.no_grad():
                smpl_target = targets_1D[b][smpl_idx].float()
                target_numi = torch.matmul(smpl_target, smpl_target.permute(1,0))
                target_tmp  = smpl_target.pow(2).sum(axis=1).pow(0.5)
                target_demi = torch.matmul(target_tmp[:,None], target_tmp[None,:])
                pi_target   = ((target_numi/target_demi)<0.5).float()

            if weights is not None:
                smpl_weight = weight_1D[b][smpl_idx, :]
                pi_obj_wght = smpl_weight + smpl_weight.permute(1,0)
            elif self.smpl_wght_en>0.0:
                pi_obj_wght = (smpl_wght + smpl_wght.permute(1,0))
            else:
                pi_obj_wght = None

            # compute loss
            if self.loss_type == 'l1':
                loss = pi_l1_loss(pi_pred, pi_target, self.margin, self.pos_wght)
            else: # 'DM-exp'
                loss = pi_exp_loss(pi_pred, pi_target, self.margin, self.pos_wght)

            if pi_obj_wght is not None:
                loss = torch.mul(loss, pi_obj_wght)

            flag = (loss>sigma).float()
            loss = (loss*flag).sum()/(flag.sum()+1.)
            all_loss.append(loss)

            # for evaluation:
            eval_pi0.append((pi_pred*(pi_target==0).float()).mean())
            eval_pi1.append((pi_pred*(pi_target==1).float()).mean())
        if len(all_loss)>0:
            return {'loss': torch.stack(all_loss).mean(),
                    'eval_pi0': torch.stack(eval_pi0).mean(),
                    'eval_pi1': torch.stack(eval_pi1).mean()}
        else:
            return None

