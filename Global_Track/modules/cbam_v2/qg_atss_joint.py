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
        self.bbox_head_track = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        # build modulators
        # self.atss_modulator = TTF_Modulator(strides=[8, 16, 32, 64],featmap_num=5)
        self.atss_modulator = Cls_Modulator(strides=[8, 16, 32, 64],featmap_num=5)
        # initialize weights
        self.atss_modulator.init_weights()
        # for classifier
        self.emb_dim = 256
        self.mid_dim = 512
        self.id_dim = 800  # 800+1
        # self.IDnet = nn.ModuleList([
        #     nn.Linear(self.emb_dim, self.mid_dim),
        #     nn.Linear(self.mid_dim, self.id_dim)
        #     ])
        self.IDnet = nn.ModuleList([
            nn.Linear(self.emb_dim, self.id_dim)
        ])
        self.loss = nn.CrossEntropyLoss()


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
                      gt_x_cls=None
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
        for embeddings, x_ij_track, x_ij_shot, i, j in self.atss_modulator(z,x, gt_bboxes_z, track_flag=True, joint_flag=True):   #　x_ij是一个特征层数个元素的列表,
            # above: select the j-th bbox/meta/label of the i-th image
            losses_ij = {}

            # print(len(x_ij))
            # print(x_ij[0].shape)
            # print(x_ij[1].shape)
            # print(x_ij[2].shape)
            # print(x_ij[3].shape)
            # print(x_ij[4].shape)

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
            for kk in range(5):
                atss_losses_ij['loss_cls'][kk] = atss_losses_ij['loss_cls'][kk]
                atss_losses_ij['loss_bbox'][kk] = atss_losses_ij['loss_bbox'][kk]
                atss_losses_ij['loss_centerness'][kk] = atss_losses_ij['loss_centerness'][kk]
            losses_ij.update(atss_losses_ij)  # not update for pure classification

            ''' for one-shot supervision '''
            gt_z_cls_ij = gt_z_cls[i][j]
            gt_bboxes_ij_shot = gt_bboxes_x[i:i + 1]
            # print(gt_bboxes_ij_shot)
            gt_bboxes_ij_shot[0] = gt_bboxes_ij_shot[0][gt_x_cls[i] == gt_z_cls_ij]
            # print(gt_x_cls[i] == gt_z_cls_ij)
            # print(gt_bboxes_ij_shot)
            gt_labels_ij_shot = gt_labels[i:i + 1]
            gt_labels_ij_shot[0] = gt_labels_ij_shot[0][gt_x_cls[i] == gt_z_cls_ij]
            img_meta_xi_shot = img_meta_x[i:i + 1]


            atss_outs_shot = self.bbox_head(x_ij_shot)
            atss_loss_inputs_shot = atss_outs_shot + (
                gt_bboxes_ij_shot, gt_labels_ij_shot, img_meta_xi_shot, self.train_cfg)
            atss_losses_ij_shot = self.bbox_head.loss(
                *atss_loss_inputs_shot,
                gt_bboxes_ignore=gt_bboxes_ignore)
            # print(atss_losses_ij_shot)
            for kk in range(5):
                atss_losses_ij_shot['loss_cls'][kk] = atss_losses_ij_shot['loss_cls'][kk]   #*0.2
                atss_losses_ij_shot['loss_bbox'][kk] = atss_losses_ij_shot['loss_bbox'][kk]  #*0.2
                atss_losses_ij_shot['loss_centerness'][kk] = atss_losses_ij_shot['loss_centerness'][kk]  #*0.2
            atss_losses_ij_shot_new = dict(
                loss_shot_cls=atss_losses_ij_shot['loss_cls'],
                loss_shot_bbox=atss_losses_ij_shot['loss_bbox'],
                loss_shot_centerness=atss_losses_ij_shot['loss_centerness'],
            )
            # print(atss_losses_ij_shot_new)

            # print(atss_losses_ij)
            # print(atss_losses_ij_shot)
            # for kk in range(5):
            #     atss_losses_ij['loss_cls'][kk] += atss_losses_ij_shot['loss_cls'][kk]*0
            #     atss_losses_ij['loss_bbox'][kk] += atss_losses_ij_shot['loss_bbox'][kk]*0
            #     atss_losses_ij['loss_centerness'][kk] += atss_losses_ij_shot['loss_centerness'][kk]*0
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
        embeddings, atss_feats_track, atss_feats_shot = next(self.atss_modulator(
            self._query, x, self._gt_bboxes_z, track_flag=True, joint_flag=True))[0:3]
        # times[1] = time.time() - begin

        # box head forward
        # get predictions
        img_shape = img_meta_x[0]['img_shape']
        scale_factor = img_meta_x[0]['scale_factor']
        img_metas = [{'img_shape': img_shape, 'scale_factor': scale_factor}]

        # begin = time.time()
        outs = self.bbox_head_track(atss_feats_track)  # out 是 cls_score, bbox_pred, centernes
        # outs_shot = self.bbox_head(atss_feats_shot)  # out 是 cls_score, bbox_pred, centernes
        # times[2] = time.time() - begin

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
