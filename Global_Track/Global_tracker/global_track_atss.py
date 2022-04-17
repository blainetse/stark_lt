import torch
import numpy as np

# from .._submodules.neuron.neuron.models import Tracker, OxUvA_Tracker
# from .._submodules.neuron.neuron.models import TrackerVOT
from neuron.models import Tracker, OxUvA_Tracker
from neuron.models import TrackerVOT
#　测试不同的数据集时，只需要将继承的父类更换一下就可以
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.core import wrap_fp16_model
from .iou import bbox_overlaps

import time

__all__ = ['GlobalATSSTrack']


class GlobalATSSTrack(Tracker):

    # def __init__(self, cfg_file, ckp_file,ckp_file_det, transforms, name_suffix=''):
    def __init__(self, cfg_file, ckp_file, transforms, name_suffix=''):
        name = 'GlobalATSSTrack'
        if name_suffix:
            name += '_' + name_suffix
        super(GlobalATSSTrack, self).__init__(
            name=name, is_deterministic=True)
        self.transforms = transforms

        # build config
        cfg = Config.fromfile(cfg_file)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        self.cfg = cfg

        # build model
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        # 第一遍加载detection模型
        # checkpoint = load_checkpoint(
        #     model, ckp_file_det, map_location='cpu')
        #
        checkpoint = load_checkpoint(
            model, ckp_file, map_location='cpu')
        model.CLASSES = ('object',)

        # GPU usage
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.model = model.to(self.device)

    @torch.no_grad()
    def init(self, img, bbox):
        self.model.eval()

        # prepare query data
        img_meta = {'ori_shape': img.shape}
        bboxes = np.expand_dims(bbox, axis=0)
        img, img_meta, bboxes = \
            self.transforms._process_query(img, img_meta, bboxes)
        img = img.unsqueeze(0).contiguous().to(
            self.device, non_blocking=True)
        bboxes = bboxes.to(self.device, non_blocking=True)

        # initialize the modulator
        begin = time.time()
        self.model._process_query(img, [bboxes])
        times = time.time() - begin

        # prepare detection module
        # self.model._process_gallary_first(
        #     img, [img_meta],bbox, rescale=True)  # by lc
        return times

    @torch.no_grad()
    def update(self, img,frame_num,threshold=0.1, **kwargs):
        self.model.eval()

        # prepare gallary data
        img_meta = {'ori_shape': img.shape}
        # begin = time.time()
        img, img_meta, _ = \
            self.transforms._process_gallary(img, img_meta, None)
        # end1 = time.time()-begin
        img = img.unsqueeze(0).contiguous().to(
            self.device, non_blocking=True)

        # get detections
        results = self.model._process_gallary(
            img, [img_meta], rescale=True, **kwargs)[0]  # by lc
        # begin = time.time()
        # results  = self.model._process_gallary(
        #     img, [img_meta],frame_num, rescale=True, **kwargs)[0]  # by lc
        # end2 = time.time() - begin
        # attention = self.model._process_gallary(
        #     img, [img_meta], rescale=True, **kwargs)[1][0]

        # if not kwargs.get('return_all', False):
        #     # return the top-1 detection
        #     max_ind = results[:, -1].argmax()
        #     # return results[max_ind, :4]    # 不是返回所有的，就是返回最大值
        #     return results[max_ind, :4], results[max_ind, 4]    # 如果是vot 18lt
        # else:
        #     # return all detections
        #     return results
        # results_nms = self.nms(results[:,:5],0.00001)  # 0.5   # =0等价于只取一个max的
        # results_nms = self.nms(results[:,:5],0)  # 0.5   # =0等价于只取一个max的
        # results_nms = results  # 0.5
        # print(results_nms.shape)
        # return results_nms
        # return results
        # results_nms = self.nms(results[:, :5], 0.5)  # 0.5 1.0
        # results = torch.from_numpy(results)
        results_nms = self.nms(results[:, :5], 0.5)  # 0.5 1.0
        indices = results_nms[:,4]>threshold  # 0.01
        return results_nms[indices]
    def compute_iou(self,box, boxes):
        """
        计算给定 box 和 boxes 之间的 IoU.
        box: 1D vector [x1, y1, x2, y2]
        boxes: [boxes_count, (x1, y1, x2, y2)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.
        Note: the areas are passed in rather than calculated here for
              efficency. Calculate once in the caller to avoid duplicate work.
        """
        # Calculate intersection areas
        x1 = np.maximum(box[0], boxes[:, 0])
        x2 = np.minimum(box[2], boxes[:, 2])
        y1 = np.maximum(box[1], boxes[:, 1])
        y2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        boxes_area = (boxes[:,2] - boxes[:,0] + 1) * (boxes[:,3] - boxes[:,1] + 1)
        union = box_area + boxes_area[:] - intersection[:]
        iou = intersection / union
        return iou

    def _compute_iou(self, boxA, boxB):  # x1,y1,w,h
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

    def nms(self, bounding_boxes, Nt):
        if len(bounding_boxes) == 0:
            return [], []
        bboxes = np.array(bounding_boxes)

        # 计算 n 个候选框的面积大小
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        scores = bboxes[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 对置信度进行排序, 获取排序后的下标序号, argsort 默认从小到大排序
        order = np.argsort(scores)

        picked_boxes = []  # 返回值
        while order.size > 0:
            # 将当前置信度最大的框加入返回值列表中
            index = order[-1]
            picked_boxes.append(bounding_boxes[index])

            # 获取当前置信度最大的候选框与其他任意候选框的相交面积
            x11 = np.maximum(x1[index], x1[order[:-1]])
            y11 = np.maximum(y1[index], y1[order[:-1]])
            x22 = np.minimum(x2[index], x2[order[:-1]])
            y22 = np.minimum(y2[index], y2[order[:-1]])
            w = np.maximum(0.0, x22 - x11 + 1)
            h = np.maximum(0.0, y22 - y11 + 1)
            intersection = w * h

            # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
            ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
            left = np.where(ious <= Nt)
            order = order[left]

        return np.array(picked_boxes)

    # def nms(self, bboxes, scores, threshold=0.5):
    #     x1 = bboxes[:, 0]
    #     y1 = bboxes[:, 1]
    #     x2 = bboxes[:, 2]
    #     y2 = bboxes[:, 3]
    #     areas = (x2 - x1) * (y2 - y1)  # [N,] 每个bbox的面积
    #     _, order = scores.sort(0, descending=True)  # 降序排列
    #     # order = np.argsort(scores)[::-1]
    #     # order = torch.Tensor(order) # 降序排列
    #
    #     keep = []
    #     while order.numel() > 0:  # torch.numel()返回张量元素个数
    #         if order.numel() == 1:  # 保留框只剩一个
    #             i = order.item()
    #             keep.append(i)
    #             break
    #         else:
    #             i = order[0].item()  # 保留scores最大的那个框box[i]
    #             keep.append(i)
    #
    #         # 计算box[i]与其余各框的IOU(思路很好)
    #         xx1 = x1[order[1:]].clamp(min=x1[i])  # [N-1,]
    #         yy1 = y1[order[1:]].clamp(min=y1[i])
    #         xx2 = x2[order[1:]].clamp(max=x2[i])
    #         yy2 = y2[order[1:]].clamp(max=y2[i])
    #         inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]
    #
    #         iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
    #         idx = (iou <= threshold).nonzero().squeeze()  # 注意此时idx为[N-1,] 而order为[N,]
    #         if idx.numel() == 0:
    #             break
    #         order = order[idx + 1]  # 修补索引之间的差值
    #     return torch.LongTensor(keep)  # Py

    def fast_nms(self,multi_bboxes,
                     multi_scores,
                     multi_coeffs,
                     score_thr,
                     iou_thr,
                     top_k,
                     max_num=-1):
        """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

            Fast NMS allows already-removed detections to suppress other detections so
            that every instance can be decided to be kept or discarded in parallel,
            which is not possible in traditional NMS. This relaxation allows us to
            implement Fast NMS entirely in standard GPU-accelerated matrix operations.

            Args:
                multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
                multi_scores (Tensor): shape (n, #class+1), where the last column
                    contains scores of the background class, but this will be ignored.
                multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
                score_thr (float): bbox threshold, bboxes with scores lower than it
                    will not be considered.
                iou_thr (float): IoU threshold to be considered as conflicted.
                top_k (int): if there are more than top_k bboxes before NMS,
                    only top top_k will be kept.
                max_num (int): if there are more than max_num bboxes after NMS,
                    only top max_num will be kept. If -1, keep all the bboxes.
                    Default: -1.

            Returns:
                tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
                    and (k, coeffs_dim). Labels are 0-based.
        """

        scores = multi_scores[:, :-1].t()  # [#class, n]
        scores, idx = scores.sort(1, descending=True)

        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]  # [#class, topk]
        num_classes, num_dets = idx.size()
        boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = iou_max <= iou_thr

        # Second thresholding introduces 0.2 mAP gain at negligible time cost
        keep *= scores > score_thr

        # Assign each kept detection to its corresponding class
        classes = torch.arange(
                num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        coeffs = coeffs[keep]
        scores = scores[keep]

        # Only keep the top max_num highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        if max_num > 0:
            idx = idx[:max_num]
            scores = scores[:max_num]

        classes = classes[idx]
        boxes = boxes[idx]
        coeffs = coeffs[idx]

        cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
        return cls_dets, classes, coeffs

        # ''' one-shot '''
        # topk = 3
        # indices = results[:, -1].argsort()[::-1][0:topk]
        # top = results[indices, :4]  # (x1,y1,x2,y2) (20,4)
        # return top, results[indices, 4]

        # topk = 100
        # indices = results[:, -1].argsort()[::-1][0:topk]
        # top = results[indices, :4]  # (x1,y1,x2,y2) (20,4)
        # ious = bbox_overlaps(np.array([gt]), top)  # (1,20)
        # # k = 1, 5 ,10, 20, 50 100
        # recall = []
        # for i in [0,4,9,19,49,99]:
        #     ious_s = ious[:,:i+1]
        #     max_ind_s = ious_s[0, :].argmax()
        #     recall_s = ious_s[0, max_ind_s]
        #     recall.append(recall_s)
        # # return the top-1 detection
        # max_ind_bb = results[:, -1].argmax()
        # return results[max_ind_bb, :4], np.array(recall)
