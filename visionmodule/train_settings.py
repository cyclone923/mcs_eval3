def getDefaultSetting():
    option = {
              'binary_alpha': 10., #2.,
              'binary_margin':1.0,
              'binary_loss_type': 'l1', # 'l1'|'CE'

              'pi_margin': 1.0,
              'pi_smpl_pairs': 5120,
              'pi_smpl_wght_en': 1,
              'pi_pos_wght': 3.0,
              'pi_loss_type': 'l1', #'l1'| 'DM-exp'
              'pi_alpha': 0.2,
              'pi_hasBG': 1,

              'regul_alpha': 0.1,
              'iou_alpha': 1.0,

              'cls_en': 1,
              'cls_iou_alpha':1.0,
              'cls_cls_alpha':1.0,
              'cls_pos_iou_thr': 0.2,

              'eval_en': 1,
              'eval_size_thrs':1.0,
              'eval_cls_score_thr': 0.5,
              'eval_iou_thr':0.5,
              'eval_classes': None,  # eval classes on classify branch = cls-fg_st_CH+1

              'model_firstLayer_en': True,
              'model_lastLayer_en': True,
              'model_clsLayer_en': True,

              'bkb_lr_alpha': 1.0,
              'fpn_lr_alpha': 1.0,
              'proto_net_lr_alpha': 1.0,
              'cls_lr_alpha':1.0,
              'iou_lr_alpha':1.0,

              'fpn_scales': [[0, 2**11], [2**10, 2**13], [2**12, 2**15], [2**14, 2**17], [2**16, 2**20]]
            }
    # fpn_scales are [d_2, d_4, d_8, d_16, d_32]


    return option
