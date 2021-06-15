from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules.loss_main import LossEvaluate
from layers.output_utils import undo_image_transformation
from dvis_network import DVIS
from train_settings import getDefaultSetting

import os
import time
import math, random
import numpy as np
import argparse
import datetime
from pathlib import Path
from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from tensorboardX import SummaryWriter


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='DVIS Training Script')
    parser.add_argument('--scripts', default=None, type=str,
                        help='json scripts to overwrite partial args')
    parser.add_argument('--batch_size', default=4, type=int,  # more sampling points
                        help='Batch size for training')
    parser.add_argument('--show_gradients', default=False, type=str2bool,
                        help='illustrate gradient on output')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                             ', the model will resume training from the interrupt file.')
    parser.add_argument('--start_iter', default=-1, type=int,
                        help='Resume training at this iter. If this is -1, the iteration will be'\
                             'determined from the file name.')
    parser.add_argument('--num_workers', default=3, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                        help='Initial learning rate. Leave as None to read this from the config.')
    parser.add_argument('--momentum', default=None, type=float,
                        help='Momentum for SGD. Leave as None to read this from the config.')
    parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                        help='Weight decay for SGD. Leave as None to read this from the config.')
    parser.add_argument('--gamma', default=None, type=float,
                        help='For each lr step, what to multiply the lr by. \
                                Leave as None to read this from the config.')
    parser.add_argument('--save_folder', default='../weights/',
                        help='Directory for saving checkpoint models.')
    parser.add_argument('--log_folder', default='../logs/',
                        help='Directory for saving logs.')
    parser.add_argument('--exp_name', default=None,
                        help='experiment to test sub-modules')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--save_interval', default=10000, type=int,
                        help='The number of iterations between saving the model.')
    parser.add_argument('--validation_size', default=5000, type=int,
                        help='The number of images to use for validation.')
    parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                        help='Only keep the latest checkpoint instead of each one.')
    parser.add_argument('--keep_latest_interval', default=100000, type=int,
                        help='When --keep_latest is on, don\'t delete the latest file at these\
                                intervals. This should be a multiple of save_interval or 0.')
    parser.add_argument('--dataset', default='coco2017_dataset', type=str,
                        help='If specified, override the dataset specified in the config with\
                                this one (example: coco2017_dataset).')
    parser.add_argument('--no_log', dest='log', action='store_false',
                        help='Don\'t log per iteration information into log_folder.')
    parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                        help='Include GPU information in the logs. Nvidia-smi tends to be slow,\
                                so set this with caution.')
    parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                        help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
    parser.add_argument('--batch_alloc', default=None, type=str,
                        help='If using multiple GPUS, you can set this to be a comma separated\
                                list detailing which GPUs should get what local batch size (It\
                                should add up to your total batch size).')
    parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                        help='DVIS will automatically scale the lr and the number of iterations\
                            depending on the batch size. Set this if you want to disable that.')

    parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
    return parser.parse_args(argv)



# Update training parameters from the config if necessary
def replace(names, args, cfg):
    for name in names:
        if getattr(args, name) == None:
            setattr(args, name, getattr(cfg, name))

class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net:DVIS, criterion:LossEvaluate):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(preds['proto'], masks,
                                preds['cls'],  targets)
        losses['rgb'] = images

        return losses

class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out

def plot_tfboard_figure(cfg, vis_imgs, vis_show=False, show_grad=False, max_vis=3):
    '''
    @Param: vis_imgs: a dict continaing,
                rgb -- [bs, 3, ht, wd]
                gts -- [bs, 1, ht, wd]
                wghts -- [bs, 1, ht, wd]
                preds -- list of tensor [bs, 1, h', w'], with h/w down_x2 one-by-one
                grad -- [bs, ch, ht, wd]
    '''
    # select sample to visual from batch
    bs = vis_imgs['gts'].size(0)
    vis_idxs = np.random.choice(bs, min(max_vis, bs), replace=False)

    # show one sample in one line
    num_col  = len(vis_imgs.keys())+1 if show_grad else len(vis_imgs.keys())
    num_col += len(vis_imgs['preds']) - 1
    fig, ax  = plt.subplots(max(2, len(vis_idxs)), num_col)
    for k, ik in enumerate(vis_idxs):
        img = undo_image_transformation(cfg.backbone, vis_imgs['rgb'][ik]) #[ht, wd, 3]
        ax[k, 0].imshow(img)

        tg  = vis_imgs['gts'][ik, 0].cpu().detach().numpy() #[ch, ht, wd]
        ax[k, 1].imshow(tg)

        tw  = vis_imgs['wghts'][ik,0].cpu().detach().numpy() #[ht, wd]
        ax[k, 2].imshow(tw)
        #plt.colorbar(ax[k, 2].imshow(tw), ax=ax[k, 2])

        sk = 3
        if show_grad:
            grad_ch = 0
            tpg     = vis_imgs['grad'][ik, grad_ch].cpu().detach().numpy()
            plt.colorbar(ax[k, 5].imshow(tpg), ax=ax[k, 3])
            sk      = 4

        for j in range(len(vis_imgs['preds'])):
            tp  = vis_imgs['preds'][j][ik,0].cpu().detach().numpy() # [ht, wd]
            plt.colorbar(ax[k, sk+j].imshow(tp), ax=ax[k, sk+j])

        # close axis
        for i in range(num_col):
            ax[k,i].axis('off')
            ax[k,i].tick_params(axis='both', left=False, top=False,
                                right=False, bottom=False,
                                labelright=False, labelbottom=False)
    if vis_show:
        plt.show()
    return fig


def train(args, cfg, option, DataSet):
    if args.exp_name is not None:
        args.save_folder = os.path.join(args.save_folder, args.exp_name)
        args.log_folder  = os.path.join(args.log_folder, args.exp_name)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder, exist_ok=True)

    # import pdb; pdb.set_trace()
    if True:
        
        dataset = DataSet(image_path=cfg.dataset.train_images,
                                mask_out_ch=cfg.gt_inst_ch,
                                info_file=cfg.dataset.train_info,
                                option = cfg.dataset,
                                transform=SSDAugmentation(cfg, MEANS),
                                running_mode='train')
    else:
        dataset = DataSet(image_path=cfg.dataset.valid_images,
                                    mask_out_ch=cfg.gt_inst_ch,
                                    info_file=cfg.dataset.valid_info,
                                    option = cfg.dataset,
                                    transform=SSDAugmentation(cfg, MEANS),
                                    running_mode='train')

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    dvis_net = DVIS(cfg)
    net = dvis_net

    net.train()
    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        dvis_net.load_weights(args.resume,
                              load_firstLayer=option['model_firstLayer_en'],
                              load_lastLayer=option['model_lastLayer_en'],
                              load_clsLayer=option['model_clsLayer_en'])

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        dvis_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)

    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                      weight_decay=args.decay)
    param_grp = [{'params': net.backbone.parameters(), 'lr':args.lr*option['bkb_lr_alpha']},
                 {'params': net.fpn.parameters(), 'lr':args.lr*option['fpn_lr_alpha']},
                 {'params': net.proto_net.parameters(), 'lr':args.lr*option['proto_net_lr_alpha']}]
    if cfg.classify_en:
        param_grp.append({'params': net.cls_convs.parameters(), 'lr': args.lr*option['cls_lr_alpha']})
        param_grp.append({'params': net.cls_linear.parameters(), 'lr': args.lr*option['cls_lr_alpha']})
        param_grp.append({'params': net.iou_linear.parameters(), 'lr': args.lr*option['iou_lr_alpha']})
    optimizer = optim.SGD(param_grp, lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    # loss criterion
    criterion = LossEvaluate(option,
                             class_weights=cfg.dataset.sem_weights,
                             fg_stCH = cfg.dataset.sem_fg_stCH)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            err_fmt = 'Error: Batch allocation (%s) does not sum to batch size (%s).'
            print(err_fmt % (args.batch_alloc, args.batch_size))
            exit(-1)

    if False:
        from ptflops import get_model_complexity_info
        flops, params = get_model_complexity_info(net,
                                                  (3, 224, 224),
                                                  as_strings=True,
                                                  print_per_layer_stat=False)


    net = NetLoss(net, criterion)
    net = CustomDataParallel(net)
    if args.cuda:
        net = net.cuda()

    # Initialize everything
    if not cfg.freeze_bn:
        dvis_net.freeze_bn() # Freeze bn so we don't kill our means

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0
    data_loader = torch.utils.data.DataLoader(dataset, args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=True,
                                              collate_fn=detection_collate,
                                              pin_memory=True)
    writer = SummaryWriter(log_dir=args.log_folder)

    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    vis_keys  = ['preds', 'gts', 'rgb', 'wghts', 'grad']
    loss_keys = ['binary', 'pi', 'iou', 'regul', 'cls_cls', 'cls_iou', \
                  'eval_prec', 'eval_rec', 'eval_acc']

    loss_avgs = { k: MovingAverage(100) for k in loss_keys }

    print('Begin training!')
    # try-except so you can use ctrl+c to save early and stop training
    try:
        log_loss  = dict()
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue

            for datum in data_loader:

                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break
                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()

                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    lr_value = min((iteration+1.) / cfg.lr_warmup_until, args.lr)
                    set_lr_value(optimizer, lr_value=lr_value)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr_scale(optimizer, lr_scale=(args.gamma ** step_index))

                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss0)
                # ta = [(key, net.state_dict(keep_vars=True)[key].requires_grad) for key in net.state_dict()]
                # net.state_dict(keep_vars=True)['module.net.refine_net.mask_fcn_logits.bias']
                ret = net(datum)

                # Mean here because Dataparallel
                losses = { k: ret[k].mean() for k in loss_keys if k in ret}
                det_loss_keys = [k for k in loss_keys if k in losses]
                all_loss = sum([losses[k] for k in det_loss_keys])
                for k in det_loss_keys:
                    loss_avgs[k].add(losses[k].item())

                # backward and optimize
                if args.show_gradients==True:
                    ret['preds_0'].retain_grad()
                    all_loss.backward(retain_graph=True)
                    ret['grad'] = ret['preds_0'].grad
                else:
                    all_loss.backward() # Do this to free up vram even if loss is not finite
                if torch.isfinite(all_loss).item():
                    optimizer.step()

                vis_imgs  = {k:ret[k] for k in vis_keys if k in ret}

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                # terminal log infor
                log_step = {'prt': 10, 'tb_scale':50, 'tb_image': 30}
                if iteration % log_step['prt'] == 0:
                    seconds=(cfg.max_iter-iteration) * time_avg.get_avg()
                    eta_str = str(datetime.timedelta(seconds=seconds)).split('.')[0]

                    total = sum([loss_avgs[k].get_avg() for k in det_loss_keys if 'eval' not in k])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_keys if k in det_loss_keys], [])

                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(det_loss_keys)) + ' T: %.3f || ETA: %s || timer: %.3f')
                            % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                # tensorboard log
                if args.log:
                    log_scale_step = log_step['tb_scale']//args.batch_size
                    for k in det_loss_keys:
                        if k not in log_loss:
                            log_loss[k] = loss_avgs[k].get_avg()
                        else:
                            log_loss[k] += loss_avgs[k].get_avg()

                    if iteration%log_scale_step == 0:
                        for k in det_loss_keys:
                            writer.add_scalar(k+'_loss',
                                              log_loss[k]/float(log_scale_step),
                                              iteration/log_scale_step)
                            log_loss[k] = 0

                    log_fig_step = log_step['tb_image']
                    if iteration%log_fig_step == 0:
                        if 'davis' in args.dataset:
                            vis_imgs['rgb'] = vis_imgs['rgb'][:, :3, :, :]
                        fig = plot_tfboard_figure(cfg, vis_imgs, show_grad=args.show_gradients)
                        writer.add_figure('grid-fig', fig, global_step=iteration/log_fig_step)

                # save ckpt
                iteration += 1
                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    dvis_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)

                # clear variables
                del ret, vis_imgs, losses, all_loss
                # end of batch run
            # end of epoch

    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')

            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)

            writer.close()
            dvis_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    writer.close()
    dvis_net.save_weights(save_path(epoch, iteration))

def set_lr_value(optimizer, lr_value=1e-4):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0:
            param_group['lr'] = lr_value

def set_lr_scale(optimizer, lr_scale=1.0):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*lr_scale

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less

        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)

        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = torch.stack(masks[cur_idx:cur_idx+alloc], dim=0)
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion, loss_keys):
    with torch.no_grad():
        losses = {}

        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())

            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break

        for k in losses:
            losses[k] /= iterations


        loss_labels = sum([[k, losses[k]] for k in loss_keys if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)


if __name__ == '__main__':
    if torch.cuda.device_count() == 0:
        print('No GPUs detected. Exiting...')
        exit(-1)

    # prepare initial parameters
    args = parse_args()
    if args.scripts is not None:
        json_dict = overwrite_args_from_json(args.scripts, args)

    # preprocess about parameters
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # import dataset related functions and configuration
    DataSet, cfg, set_dataset, set_cfg = dataset_specific_import(args.dataset)

    # update config and args
    if args.config is not None:
        set_cfg(cfg, args.config)
    if args.dataset is not None:
        set_dataset(cfg, args.dataset)

    if args.batch_size // torch.cuda.device_count() < 6:
        print('Per-GPU batch size is less than the recommended \
                            limit for batch norm. Disabling batch norm.')
        cfg.freeze_bn = True

    if torch.cuda.device_count()>1:
        args.show_gradients = False

    if args.autoscale and args.batch_size != 8:
        factor = args.batch_size / 8
        print('Scaling parameters by %.2f for a bsize of %d.' % (factor, args.batch_size))
        cfg.lr *= factor
        cfg.max_iter //= factor
        cfg.lr_steps = [x // factor for x in cfg.lr_steps]
    replace(['lr', 'decay', 'gamma', 'momentum'], args, cfg)

    # extra parameters defined in json scripts
    # net_arch is 'SC' | 'MC-ROI' | 'MC-noROI'
    option = getDefaultSetting()
    if args.scripts is not None:
        overwrite_params_from_json(json_dict, option)

    # train the network
    train(args, cfg, option, DataSet)
