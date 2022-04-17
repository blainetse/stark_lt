from mmdet.core import bbox2result
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn as nn
import time

from mmdet.models.registry import DETECTORS
from mmdet.models.detectors.base import BaseDetector
from mmdet.core import auto_fp16, get_classes, tensor2imgs, \
    bbox2result, bbox2roi, build_assigner, build_sampler

from .modulators import RPN_Modulator, Cls_Modulator

from mmdet.models import builder

# ttf的策略是，可以在backbone处加入调制信号，这样上采样和short cut融合都会从中受益，还是先采取globaltrack的调制模式

map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 60, 11: 60, 12: 60, 13: 10, 14: 11, 15: 12, 16: 13,
            17: 14, 18: 15, 19: 16, 20: 17, 21: 60, 22: 18, 23: 19, 24: 20, 25: 21, 26: 22, 27: 23, 28: 24, 29: 60, 30: 25, 31: 60,
            32: 26, 33: 27, 34: 60, 35: 60, 36: 28, 37: 29, 38: 30, 39: 31, 40: 32, 41: 33, 42: 34, 43: 35, 44: 36, 45: 37, 46: 38,
            47: 39, 48: 60, 49: 40, 50: 41, 51: 42, 52: 43, 53: 44, 54: 45, 55: 46, 56: 47, 57: 48, 58: 49, 59: 60, 60: 50, 61: 60,
            62: 51, 63: 52, 64: 60, 65: 53, 66: 60, 67: 54, 68: 60, 69: 60, 70: 60, 71: 55, 72: 60, 73: 56, 74: 57, 75: 58, 76: 60, 77: 59, 78: 60, 79: 60}

div_map_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 60: 79, 10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18,
                16: 19, 17: 20, 18: 22, 19: 23, 20: 24, 21: 25, 22: 26, 23: 27, 24: 28, 25: 30, 26: 32, 27: 33, 28: 36, 29: 37, 30: 38,
                31: 39, 32: 40, 33: 41, 34: 42, 35: 43, 36: 44, 37: 45, 38: 46, 39: 47, 40: 49, 41: 50, 42: 51, 43: 52, 44: 53, 45: 54,
                46: 55, 47: 56, 48: 57, 49: 58, 50: 60, 51: 62, 52: 63, 53: 65, 54: 67, 55: 71, 56: 73, 57: 74, 58: 75, 59: 77}

__all__ = ['QG_ATSS_JOINT']

@DETECTORS.register_module
class QG_ATSS_JOINT(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 bbox_head_track=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(QG_ATSS_JOINT, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
            # for para in self.neck.parameters():  # by lc
            #     para.requires_grad = False
        self.bbox_head = builder.build_head(bbox_head)
        self.bbox_head_track = builder.build_head(bbox_head_track)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        # build modulators
        # self.atss_modulator = TTF_Modulator(strides=[8, 16, 32, 64],featmap_num=5)
        self.atss_modulator = Cls_Modulator(strides=[8, 16, 32, 64],featmap_num=4)
        # initialize weights
        self.atss_modulator.init_weights()
        # for classifier
        self.emb_dim = 256
        self.mid_dim = 120 #512
        self.id_dim = 80+1   # 80+'others'     # 800+1
        self.id_dim_new = 61
        self.id_dim_shot = 220
        # self.IDnet = nn.ModuleList([
        #     nn.Linear(self.emb_dim, self.mid_dim),
        #     nn.Linear(self.mid_dim, self.id_dim)
        #     ])
        self.IDnet = nn.ModuleList([
            nn.Linear(self.emb_dim, self.id_dim)
        ])
        self.loss = nn.CrossEntropyLoss()

        self.mask_loss = nn.BCELoss()
        self.weight_shot = nn.Parameter(torch.Tensor([0.5]))
        self.weight_track = nn.Parameter(torch.Tensor([0.5]))
        self.mask_loss = nn.BCELoss()

    def init_weights(self, pretrained=None):
        super(QG_ATSS_JOINT, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()
        self.bbox_head_track.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)  # len = 4
        # print(np.any(np.isnan(x[0].cpu().detach().numpy())))
        if self.with_neck:
            x = self.neck(x)
        return x

    @auto_fp16(apply_to=('img_z', 'img_x'))
    def forward(self,
                img_z,
                img_x,
                img_meta_z,
                img_meta_x,
                return_loss=True,
                **kwargs):
        if return_loss:
            return self.forward_train(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)
        else:
            return self.forward_test(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img_z,
                      img_x,
                      img_meta_z,
                      img_meta_x,
                      gt_bboxes_z,
                      gt_bboxes_x,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_z_cls=None,
                      gt_x_cls=None,
                      gt_x_cls_shot = None,
                      gt_bboxes_x_shot = None,
                      gt_bboxes_z_shot = None,
                      gt_z_cls_shot = None,
                      gt_labels_shot = None,
                      data_set=None
                      ):
        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)

        # print(img_z.shape)
        # print(img_x.shape)
        # print(gt_z_cls)
        # for para in self.backbone.parameters():  # all of is none
        #     if para.requires_grad:
        #         print(para)
        #
        # for para in self.neck.parameters():  # all of is none
        #     if para.requires_grad:
        #         print(para)

        # for para in self.atss_modulator.parameters():
        #     if para.requires_grad:
        #         print(para)

        losses = {}
        total = 0.
        x_ij_track_list = [[] for _ in range(z[0].shape[0])]  # 外面一层是i=1的意思
        gt_mask_zero = torch.zeros((1, 1, img_x.shape[2], img_x.shape[3]))
        for embeddings,att, x_ij_track, x_ij_shot, i, j in self.atss_modulator(self.IDnet,z[:4],x[:4], gt_bboxes_z,gt_z_cls, track_flag=True, joint_flag=True):   #　x_ij是一个特征层数个元素的列表,
            # above: select the j-th bbox/meta/label of the i-th image
            x_ij_track_list[i].append(x_ij_track)  # 存储特征池，以便后续挑选负样本特征
            num = len(x_ij_track_list[i])
            losses_ij = {}


            ''' select the j-th bbox/meta/label of the i-th image '''
            gt_bboxes_ij = gt_bboxes_x[i:i + 1]
            gt_bboxes_ij[0] = gt_bboxes_ij[0][j:j + 1]
            gt_labels_ij = gt_labels[i:i + 1]
            gt_labels_ij[0] = gt_labels_ij[0][j:j + 1]
            img_meta_xi = img_meta_x[i:i + 1]


            # gt_bboxes_ij = gt_bboxes_x[i:i + 1]
            # gt_labels_ij = gt_labels[i:i + 1]
            # img_meta_xi = img_meta_x[i:i + 1]
            # print(gt_bboxes_ij)
            # print(gt_labels_ij)
            ''' end '''

            # ATSS forward and loss atss head是从backbone后面开始，包括上采样在内的
            atss_outs = self.bbox_head_track(x_ij_track)  # out 是 cls_score, bbox_pred
            atss_loss_inputs = atss_outs + (
                gt_bboxes_ij, gt_labels_ij, img_meta_xi, self.train_cfg)
            atss_losses_ij = self.bbox_head_track.loss(
                *atss_loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore)  # atss_losses_ij：  dict(loss_cls=losses_cls, loss_bbox=losses_bbox不止)

            ''' for negative exampler'''
            if num > 1:
                choice = np.random.randint(low=0,high=num-1)
                x_ij_ne_track = x_ij_track_list[i][choice]
                atss_outs_ne = self.bbox_head_track(x_ij_ne_track)  # out 是 cls_score, bbox_pred
                atss_losses_ij_ne = self.bbox_head_track.loss_cls_score(
                    atss_outs_ne[0], self.train_cfg)  # atss_losses_ij：  dict(loss_cls=losses_cls, loss_bbox=losses_bbox不止)
                for kk in range(4):
                    atss_losses_ij['loss_cls'][kk] = (atss_losses_ij['loss_cls'][kk] + atss_losses_ij_ne['loss_cls'][kk])/2

            losses_ij.update(atss_losses_ij)  # not update for pure classification

            ''' for one-shot supervision '''
            # gt_bboxes_ij_shot = gt_bboxes_x_shot[i:i + 1]
            # # gt_labels_ij_shot = gt_labels_shot[i:i + 1]
            # img_meta_xi_shot = img_meta_x[i:i + 1]
            # gt_labels_ij_shot = [ gt_x_cls[0][...] + 1 ]  # 只对1的batch成立
            # for k in range(len(gt_labels_ij_shot[0])):
            #     if gt_labels_ij_shot[0][k]>80:
            #         gt_labels_ij_shot[0][k] = 0
            ############[  for coco-lasot-got10k datasets psudeo labels : one shot part  ]###############
            # print(data_set)
            # if data_set[i][0]==1.0:


            # atss_outs_shot = self.bbox_head(x_ij_shot)  #torch.Size([1, 80, 96, 168])
            # atss_loss_inputs_shot = atss_outs_shot + (
            #     gt_bboxes_ij_shot, gt_labels_ij_shot, img_meta_xi_shot, self.train_cfg)
            # atss_losses_ij_shot = self.bbox_head.loss(
            #     *atss_loss_inputs_shot,
            #     gt_bboxes_ignore=gt_bboxes_ignore)
            # # print(atss_losses_ij_shot)
            # # for kk in range(4):
            # #     atss_losses_ij_shot['loss_cls'][kk] = atss_losses_ij_shot['loss_cls'][kk]*1  #*0.2
            # #     atss_losses_ij_shot['loss_bbox'][kk] = atss_losses_ij_shot['loss_bbox'][kk]*1  #*0.2
            # #     atss_losses_ij_shot['loss_centerness'][kk] = atss_losses_ij_shot['loss_centerness'][kk]*1  #*0.2
            # atss_losses_ij_shot_new = dict(
            #     loss_shot_cls=atss_losses_ij_shot['loss_cls'],
            #     loss_shot_bbox=atss_losses_ij_shot['loss_bbox'],
            #     loss_shot_centerness=atss_losses_ij_shot['loss_centerness'],
            # )

            # print(atss_losses_ij_shot_new)

            # print(atss_losses_ij)
            # print(atss_losses_ij_shot)
            # for kk in range(5):
            #     atss_losses_ij['loss_cls'][kk] += atss_losses_ij_shot['loss_cls'][kk]*0
            #     atss_losses_ij['loss_bbox'][kk] += atss_losses_ij_shot['loss_bbox'][kk]*0
            #     atss_losses_ij['loss_centerness'][kk] += atss_losses_ij_shot['loss_centerness'][kk]*0
            # losses_ij.update(atss_losses_ij_shot_new)

            # ''' for psudeo mask'''
            # gt_x_ij = gt_bboxes_x[i][j:j + 1][0]  # (x1,y1,x2,y2)  # gt_bboxes_x[i].shape[0] -> num
            # x1 = torch.floor(gt_x_ij[0]).to(torch.int32)
            # y1 = torch.floor(gt_x_ij[1]).to(torch.int32)
            # x2 = torch.ceil(gt_x_ij[2]).to(torch.int32)
            # y2 = torch.ceil(gt_x_ij[3]).to(torch.int32)
            # gt_mask = gt_mask_zero
            # gt_mask[..., y1:y2, x1:x2] = 1  # (1,1,h,w)
            # gt_mask = nn.functional.interpolate(gt_mask, size=att.shape[2:]).cuda()
            # # print(att.shape)
            # # loss_mask_ij = self.mask_loss(att[:,level,...].unsqueeze(dim=1),gt_mask)  # no argmax
            # loss_mask_ij = self.mask_loss(att, gt_mask)
            # losses_ij.update({'loss_mask_mid': [loss_mask_ij * 0.5]})

            ''' for cls '''
            self.cls_original = gt_z_cls[i][j:j + 1]
            # self.cls = self.cls_original.cuda()
            if self.cls_original[0] > 79:
                self.cls = torch.Tensor([80]).cuda() # 60  # 'others'
            else:
                self.cls = torch.Tensor([self.cls_original[0].item()]).cuda()
            self.id_target = self.cls.long()

            # self.cls_ori = gt_z_cls[i][j:j + 1]
            # self.id_target = self.cls_ori.long()

            # print(embeddings.shape)
            id_pred = self.IDnet[0](embeddings.view(1, -1))  #### 9.28
            # id_pred = self.IDnet[1](id_pred)  #12.05 12.05.xz
            # id_pred_ = self.IDnet[0](embeddings)
            # id_pred = self.IDnet[1](id_pred_.view((1, -1)))
            # print(self.id_target)
            cls_losses_ij = self.loss(id_pred, self.id_target)
            losses_ij.update({'loss_cls_mid': [cls_losses_ij*0.2]})

            ''' for psudeo mask'''
            gt_x_ij = gt_bboxes_x[i][j:j + 1][0]  # (x1,y1,x2,y2)
            x1 = torch.floor(gt_x_ij[0]).to(torch.int32)
            y1 = torch.floor(gt_x_ij[1]).to(torch.int32)
            x2 = torch.ceil(gt_x_ij[2]).to(torch.int32)
            y2 = torch.ceil(gt_x_ij[3]).to(torch.int32)
            gt_mask = gt_mask_zero
            gt_mask[..., y1:y2, x1:x2] = 1  # (1,1,h,w)
            gt_mask = nn.functional.interpolate(gt_mask, size=att.shape[2:]).cuda()
            loss_mask_ij = self.mask_loss(att, gt_mask)
            losses_ij.update({'loss_mask_mid': [loss_mask_ij * 0.5]})

            if j == 0:
                # print(gt_x_cls)
                gt_bboxes_ij_shot = gt_bboxes_x_shot  # 只对1的batch成立
                img_meta_xi_shot = img_meta_x
                gt_labels_ij_shot = [gt_x_cls_shot[0][...] + 1]
                for k in range(len(gt_labels_ij_shot[0])):
                    if gt_labels_ij_shot[0][k] > 80:
                        gt_labels_ij_shot[0][k] = 0
                ############[  for coco-lasot-got10k datasets psudeo labels : one shot part  ]###############
                # print(data_set)
                if data_set[i][0]==1.0:
                    gt_x_ij = gt_bboxes_x[i][j:j + 1][0]  # (x1,y1,x2,y2)  对于data_set=1.0,就是tracking数据集只有一个标签，所以应该是成立的
                    x1 = torch.floor(gt_x_ij[0]).to(torch.int32)
                    y1 = torch.floor(gt_x_ij[1]).to(torch.int32)
                    x2 = torch.ceil(gt_x_ij[2]).to(torch.int32)
                    y2 = torch.ceil(gt_x_ij[3]).to(torch.int32)
                    h = img_x.shape[2]
                    w = img_x.shape[3]
                    x1 = x1 if x1 > 0 else 0
                    y1 = y1 if y1 > 0 else 0
                    x2 = x2 if x2 < w else w
                    y2 = y2 if y2 < h else h
                    gt_mask = gt_mask_zero
                    gt_mask[..., y1:y2, x1:x2] = 1  # (1,1,h,w)
                    label_weight_list = [nn.functional.interpolate(gt_mask, size=x[i].shape[2:]).cuda()
                                            for i in range(len(x))]  # 1,1,H,W
                    img_meta_xi_shot[0].update({'label_weight_list': label_weight_list})
                ############[  end  ]###############

                atss_outs_shot = self.bbox_head(x)  # torch.Size([1, 80, 96, 168])
                atss_loss_inputs_shot = atss_outs_shot + (
                    gt_bboxes_ij_shot, gt_labels_ij_shot, img_meta_xi_shot, self.train_cfg)
                atss_losses_ij_shot = self.bbox_head.loss(
                    *atss_loss_inputs_shot,
                    gt_bboxes_ignore=gt_bboxes_ignore)

                atss_losses_ij_shot_new = dict(
                    loss_shot_cls=atss_losses_ij_shot['loss_cls'],
                    loss_shot_bbox=atss_losses_ij_shot['loss_bbox'],
                    loss_shot_centerness=atss_losses_ij_shot['loss_centerness'],
                )
                losses_ij.update(atss_losses_ij_shot_new)

            # update losses  就是将loss_ij添加到loss的字典中
            # print(losses_ij['loss_bbox'])
            for k, v in losses_ij.items():
                if k in losses:
                    if isinstance(v, (tuple, list)):
                        for u in range(len(v)):
                            losses[k][u] += v[u]
                    else:
                        losses[k] += v
                else:
                    losses[k] = v
            total += 1.
        # print(losses)
        # average the losses over instances
        for k, v in losses.items():
            if k.startswith('loss_shot'):
                pass
            else:
                if isinstance(v, (tuple, list)):
                    for u in range(len(v)):
                        losses[k][u] /= total
                else:
                    losses[k] /= total

        # 每个key的value是有5个元素,是代表着有5层
        # print('一张图片结束啦')
        return losses

    def forward_test(self,
                     img_z,
                     img_x,
                     img_meta_z,
                     img_meta_x,
                     gt_bboxes_z,
                     rescale=False,
                     **kwargs):
        # assume one image and one instance only
        return self.simple_test(
            img_z, img_x, img_meta_z, img_meta_x,
            gt_bboxes_z,rescale, **kwargs)

    def simple_test(self,
                    img_z,
                    img_x,
                    img_meta_z,
                    img_meta_x,
                    gt_bboxes_z,
                    rescale=False,
                    **kwargs):
        # assume one image and one instance only
        assert len(img_z) == 1
        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)

        atss_feats = next(self.atss_modulator(z, x, gt_bboxes_z))[1]  # for one-shot

        # get predictions
        img_shape = img_meta_x[0]['img_shape']
        scale_factor = img_meta_x[0]['scale_factor']
        img_metas = [{'img_shape': img_shape, 'scale_factor': scale_factor}]

        outs = self.bbox_head(atss_feats)  # out 是 cls_score, bbox_pred
        bbox_inputs = outs + (img_metas, self.test_cfg,rescale)  # 最后的参数实际是rescale = False
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)  # 在anchorhead里面，atss head是继承于anchorhead

        # 都是产生一些个，最后选择最高得分的即可

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)[0]
            for det_bboxes, det_labels in bbox_list
        ]
        return np.array(bbox_results)   # bbox_list的每个元素: det_bboxes, det_labels

    # bbox_results应该是一个list,每个元素都是一个1-list的,其中的元素是（1,,5）, 经过改动，认为得到的是（n,5)的numpy数组，和之前的two-stage保持相同的输出

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def _process_query(self, img_z, gt_bboxes_z):
        self._query = self.extract_feat(img_z)
        self._gt_bboxes_z = gt_bboxes_z

    def get_mmresult(self, img, result, dataset='coco', score_thr=0.1):
        # class_names = get_classes(dataset)

        # labels = [
        #     np.full(bbox.shape[0], i, dtype=np.int32)
        #     for i, bbox in enumerate(result[0])
        # ]
        # labels = np.concatenate(labels)
        # bboxes = np.vstack(result)
        labels = result[0][1]
        bboxes = result[0][0]
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        return bboxes, labels

    def _compute_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        if xA < xB and yA < yB:
            # compute the area of intersection rectangle
            interArea = (xB - xA) * (yB - yA)
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the intersection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
        else:
            iou = 0

        assert iou >= 0
        assert iou <= 1.01

        return iou

    def _process_gallary_first(self, img_x, img_meta_x, init_gt1,rescale=False, **kwargs):
        x = self.extract_feat(img_x)
        outs_o = self.bbox_head_det(x)  # out 是 cls_score, bbox_pred
        bbox_inputs_o = outs_o + (img_meta_x, self.test_cfg, rescale)  # 最后的参数传进来实际是rescale = True
        bbox_list_o = self.bbox_head_det.get_bboxes(*bbox_inputs_o)  # 在anchorhead里面，atss head是继承于anchorhead
        bboxes, labels = self.get_mmresult(img_x, bbox_list_o)
        boxes = bboxes[:, :4]
        mmscore = bboxes[:, -1]
        boxes = np.array(boxes.cpu())
        boxes = np.array([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
        iou = np.zeros((boxes.shape[1],))
        for i in range(boxes.shape[1]):
            iou[i] = self._compute_iou(boxes[:, i], init_gt1)
        if iou.shape[0] == 0:
            self.label = None
        elif max(iou) > 0.4 or max(iou) > 0.1 and mmscore[np.argmax(iou)] > 0.5:
            self.label = labels[np.argmax(iou)]
        else:
            self.label = None
        print(self.label)


    def _process_gallary(self, img_x, img_meta_x,rescale = False, **kwargs):  # for one-shot
        times = np.zeros(4)  # python的二维数据表示要用二层括号来进行表示
        # begin = time.time()
        x = self.extract_feat(img_x)
        # times[0] = time.time() - begin

        # atss modulator forward
        # begin = time.time()
        embeddings,att, atss_feats_track, atss_feats_shot = next(self.atss_modulator(self.IDnet,
            self._query[:4], x[:4], self._gt_bboxes_z, track_flag=True, joint_flag=True))[0:4]
        # times[1] = time.time() - begin

        # box head forward
        # get predictions
        img_shape = img_meta_x[0]['img_shape']
        scale_factor = img_meta_x[0]['scale_factor']
        img_metas = [{'img_shape': img_shape, 'scale_factor': scale_factor}]

        # begin = time.time()
        outs = self.bbox_head_track(atss_feats_track)  # out 是 cls_score, bbox_pred, centernes
        outs_shot = self.bbox_head(atss_feats_shot)  # out 是 cls_score, bbox_pred, centernes
        # times[2] = time.time() - begin

        id_pred = self.IDnet[0](embeddings.view(1, -1))
        # id_pred = self.IDnet[1](id_pred)
        id_pred_ = nn.functional.softmax(id_pred)
        # print(torch.argmax(id_pred_[0, :]).item())
        # print(max(id_pred_[0, :]).item())
        cls = torch.argmax(id_pred_[0, :]).item()
        # cls = div_map_dict[cls]
        # print(cls)
        # print(self.weight_shot)
        # print(self.weight_track)
        # if torch.argmax(id_pred_[0, :]).item() == 80 or torch.argmax(id_pred_[0, :]).item() == 60:
        #     switch_up = False
        # else:
        #     switch_up = True
        # m = nn.Sigmoid()
        # if switch_up:
        #     for kk in range(4):
        #         outs[0][kk] = outs[0][kk]*self.weight_track + outs_shot[0][kk][:, cls:cls + 1, ...]*self.weight_shot
        # import cv2
        # m = nn.Sigmoid()
        # heatmap = m(outs[0][0][0, cls:cls + 1, ...]).cpu().permute(1, 2, 0).numpy() * 255
        # size = heatmap.shape[0:2]
        # from skimage import transform
        # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 2, heatmap.shape[1] * 2))
        # heatmap0 = heatmap.astype(np.uint8)
        # heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # HOT
        # cv2.imshow('heatmap-class', heatmap)
        # cv2.waitKey(1)
        #
        # heatmap = m(outs_shot[0][1][0, cls:cls + 1, ...]).cpu().permute(1, 2, 0).numpy() * 255
        # size = heatmap.shape[0:2]
        # from skimage import transform
        # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 4, heatmap.shape[1] * 4))
        # heatmap0 = heatmap.astype(np.uint8)
        # heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # HOT
        # cv2.imshow('heatmap1-class', heatmap)
        # cv2.waitKey(1)
        #
        # heatmap = m(outs_shot[0][2][0, cls:cls + 1, ...]).cpu().permute(1, 2, 0).numpy() * 255
        # size = heatmap.shape[0:2]
        # from skimage import transform
        # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 8, heatmap.shape[1] * 8))
        # heatmap0 = heatmap.astype(np.uint8)
        # heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # HOT
        # cv2.imshow('heatmap2-class', heatmap)
        # cv2.waitKey(1)
        #
        # heatmap = m(outs_shot[0][3][0, cls:cls + 1, ...]).cpu().permute(1, 2, 0).numpy() * 255
        # size = heatmap.shape[0:2]
        # from skimage import transform
        # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 16, heatmap.shape[1] * 16))
        # heatmap0 = heatmap.astype(np.uint8)
        # heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # HOT
        # cv2.imshow('heatmap3-class', heatmap)
        # cv2.waitKey(1)

        # begin = time.time()
        bbox_inputs = outs + (img_metas, self.test_cfg, rescale)  # 最后的参数传进来实际是rescale = True
        bbox_list = self.bbox_head_track.get_bboxes(*bbox_inputs)  # 在anchorhead里面，atss head是继承于anchorhead

        # bbox_inputs_shot = outs_shot + (img_metas, self.test_cfg, rescale)  # 最后的参数传进来实际是rescale = True
        # bbox_list_shot = self.bbox_head.get_bboxes(*bbox_inputs_shot)  # 在anchorhead里面，atss head是继承于anchorhead
        # times[3] = time.time() - begin
        # print(times)
        # print('~~~~~~~~~~~~~~~~~')
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head_track.num_classes)[0]
            for det_bboxes, det_labels in bbox_list
        ]
        # bbox_results = [
        #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)[0]
        #     for det_bboxes, det_labels in bbox_list_shot
        # ]


        bbox_results = bbox_results


        return np.array(bbox_results) #, times   #　返回的是一个数组,最终会挑选得分最高的那个

