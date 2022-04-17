import torch
import numpy as np

from neuron.models import Tracker, OxUvA_Tracker
from neuron.models import TrackerVOT
#　测试不同的数据集时，只需要将继承的父类更换一下就可以
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.core import wrap_fp16_model

import time

__all__ = ['OneShotTrack']

# modified version, not initial tracking config, is diffrent !
class OneShotTrack(Tracker):

    # def __init__(self, cfg_file, ckp_file,ckp_file_det, transforms, name_suffix=''):
    def __init__(self, cfg_file, ckp_file, transforms, name_suffix=''):
        name = 'OneShotTrack'
        if name_suffix:
            name += '_' + name_suffix
        super(OneShotTrack, self).__init__(
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

        self.model.eval()

    @torch.no_grad()
    def init(self, img, bbox):
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
    def update(self, img, gt, **kwargs):
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
        # begin = time.time()
        results = self.model._process_gallary(
            img, [img_meta], rescale=True, **kwargs)[0]  # by lc 
        # end2 = time.time() - begin

        # if not kwargs.get('return_all', False):
        #     # return the top-1 detection
        #     max_ind = results[:, -1].argmax()
        #     # return results[max_ind, :4]    # 不是返回所有的，就是返回最大值
        #     return results[max_ind, :4], results[max_ind, 4]    # 如果是vot 18lt
        # else:
        #     # return all detections
        #     return results

        # one-shot
        inds = results[:, -1] > 0.10  # threshold  0.05
        results = torch.from_numpy(results[inds,:])
        # nms
        keep = nms(results[:,:4],results[:,4])
        nms_dets = results[keep, :]
        # topk = 5
        # indices = results[:, -1].argsort()[::-1][0:topk]
        # top = results[indices, :4]  # (x1,y1,x2,y2) (20,4)
        # return top, results[indices, 4]
        # return results[indices, :5]
        return nms_dets

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

# NMS算法
    # bboxes维度为[N,4]，scores维度为[N,], 均为tensor
def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor