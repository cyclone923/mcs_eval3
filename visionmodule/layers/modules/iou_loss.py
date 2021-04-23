import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    '''Adopt the Hungarian algorithm for bipartite matching based on the input cost matrix
       Implementation is inherited from the Facebook-DETR repository:
            https://github.com/facebookresearch/detr/blob/master/models/matcher.py
    '''
    def __init__(self):
        super(HungarianMatcher, self).__init__()

    @torch.no_grad()
    def forward(self, costs):
        ''' Performs the matching
        Params:
            costs: tensor in size [bs, num_queries, num_targets].

        Outputs:
            indices: list of matches.
        '''
        costs = costs.cpu()
        indices = [linear_sum_assignment(costs[i]) for i in range(costs.size()[0])]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class IoULoss(nn.Module):
    '''This class computes IoU loss based on between the targets and the predictions of the network.

    The iou-loss is computed as: target*(1-pred) + (1-target)*pred, so that:
        for target that is real target: pred need to be close to target to have less loss
        for target that is empty for batch alignment: pred need to be empty too to have less loss.
        (+ 1e0) in the denominator is also for divie-0 pretection.

    matching between targets and predictions is bipartite projection.
    '''

    def __init__(self, FG_stCH=1, compute_loss=True):
        '''
        @param: FG_stCH: started channel for FG in preds and targets
        '''
        super(IoULoss, self).__init__()

        self.FG_stCH      = FG_stCH
        self.compute_loss = compute_loss
        self.matcher      = HungarianMatcher()

    @torch.no_grad()
    def nonhungarian_match_result(self, cost):
        '''
        Params:
            cost: in size [bs, num_preds, num_targetsi], with num_preds==num_targets
        '''
        bs, ch, _  = cost.size()
        _, match_FG_indices = cost.min(axis=-1) # [bs, ch]

        # add of BG channel macher to (indices, loss) for further process.
        BG_idxs = torch.cumsum(torch.ones(self.FG_stCH, dtype=torch.long), axis=0)-1
        full_indices = []
        for b in range(bs):
            FG_pred_idxs = torch.cumsum(torch.ones(ch, dtype=torch.long), axis=0)-1
            full_indices.extend(torch.cat([BG_idxs,FG_pred_idxs+self.FG_stCH], axis=0))

            FG_target_idxs = match_FG_indices[b]
            full_indices.extend(torch.cat([BG_idxs,FG_target_idxs+self.FG_stCH], axis=0))
        full_indices = torch.stack(full_indices).view(bs, 2, -1)
        return full_indices

    @torch.no_grad()
    def hungarian_match_result(self, hung_indices):
        '''
        Params:
            hung_indices: list, match indices from the hungarian matcher
        '''
        bs  = len(hung_indices)

        # add of BG channel macher to (indices, loss) for further process.
        BG_idxs = torch.cumsum(torch.ones(self.FG_stCH, dtype=torch.long), axis=0)-1
        full_indices = []
        for b in range(bs):
            full_indices.extend(torch.cat([BG_idxs,hung_indices[b][0]+self.FG_stCH], axis=0))
            full_indices.extend(torch.cat([BG_idxs,hung_indices[b][1]+self.FG_stCH], axis=0))
        full_indices = torch.stack(full_indices).view(bs, 2, -1)
        return full_indices

    @torch.no_grad()
    def compute_real_iou_from_match(self, preds, targets, match_indices):
        '''
        Params:
            preds: prediction of the network, in size [bs, ch, N]
            targets: tensor, in size [bs, N, ch']
            match_indices: size [bs, 2, ch], with [0] for preds, [1] for targets
        '''
        # extend bg and compute real iou
        bs  = match_indices.size(0)
        iou = []
        for b in range(bs):
            for i, j in zip(match_indices[b][0], match_indices[b][1]):
                tmp_pred = (preds[b, i:i+1,:]>0.5).float() # [1, N]
                tmp_target = targets[b, :, j:j+1] #[N, 1]
                intp  = torch.matmul(tmp_pred,tmp_target)
                union = tmp_pred.sum() + tmp_target.sum() - intp
                iou.append(intp/(union+1.))
        return iou

    def forward(self, preds, targets, return_hungarian_map=True):
        '''perform bipartite matching and compute the loss

        Params:
            preds: prediction of the network, in size [bs, ch, ht, wd]
            targets: prediction of the network, in size [bs, ch', ht, wd]
            return_hungarian_map: return matches is hungarian result or not
        '''
        g_ch             = targets.size(1)
        bs, p_ch, ht, wd = preds.size()
        preds_0        = preds.view(bs, p_ch, -1).float() #[bs, ch, N]
        FG_preds       = preds_0[:, self.FG_stCH:, :] #[bs, ch, N]

        with torch.no_grad():
            targets_0      = targets.view(bs, g_ch, -1).permute([0,2,1]).float() # [bs, N, ch']
            FG_targets     = targets_0[:, :, self.FG_stCH:] #[bs, N, ch']
            FG_size    = FG_targets.sum(dim=1, keepdim=True) # [bs, 1, ch]

        # compute match cost for FG objects channels between preds and targets
        if False:
            FG_cost    = torch.matmul(1-FG_preds, FG_targets)/(FG_size+1.) # [bs, ch, ch']
            BG_cost    = torch.matmul(FG_preds, 1-FG_targets)/(FG_size+1.)
            match_cost = FG_cost + BG_cost
        else:
            FG_size_p  = FG_preds.sum(dim=2, keepdim=True)  # [bs, ch, 1]
            intp       = torch.matmul(FG_preds, FG_targets) # [bs, ch, ch']
            match_cost = 1 - (2* intp +1e-2)/(FG_size_p+FG_size + 1e-2)


        # FG_indices and full_indices from matching
        if return_hungarian_map or self.compute_loss:
            FG_indices   = self.matcher(match_cost)

        if return_hungarian_map:
            full_indices = self.hungarian_match_result(FG_indices)
        else:
            full_indices = self.nonhungarian_match_result(match_cost)
        iou = self.compute_real_iou_from_match(preds_0, targets_0, full_indices)

        ret =  {'loss': None,
                'iou-iou': torch.stack(iou).view(bs, -1),
                'indices': full_indices}

        # compute loss
        if self.compute_loss:
            # compute loss on BG categories channels.
            BG_preds   = preds_0[:, :self.FG_stCH, :] # [bs, ch, N]
            BG_targets = targets_0[:, :, :self.FG_stCH].permute([0,2,1]) # [bs, ch, N]
            BG_size    = BG_targets.sum(dim=2).float() # [bs, ch]

            FG_B_cost  = ((1-BG_preds) * BG_targets).sum(dim=-1)/(BG_size + 1.)
            BG_B_cost  = (BG_preds * (1-BG_targets)).sum(dim=-1)/(ht*wd-BG_size + 1.)
            BG_loss    = FG_B_cost + BG_B_cost # [bs, ch]

            # geometry loss on all channels (BG + FG objects)
            loss = []
            for b in range(bs):
                loss.extend(BG_loss[b, :self.FG_stCH])
                for i, j in zip(FG_indices[b][0], FG_indices[b][1]):
                    loss.append(match_cost[b, i, j])

            ret['loss'] = torch.stack(loss).mean()

        return ret

