# import torch
# import numpy as np
#
# from neuron.models import Tracker, OxUvA_Tracker
# from neuron.models import TrackerVOT
# #　测试不同的数据集时，只需要将继承的父类更换一下就可以
# from mmcv import Config
# from mmcv.runner import load_checkpoint
# from mmdet.models import build_detector
# from mmdet.core import wrap_fp16_model
# from .iou import bbox_overlaps
#
# import time
# # pymdnet
# import sys
# from pyMDNet.modules.model import *
# sys.path.insert(0, './pyMDNet')
# from pyMDNet.modules.model import MDNet, BCELoss, set_optimizer
# from pyMDNet.modules.sample_generator import SampleGenerator
# from pyMDNet.modules.utils import overlap_ratio
# from pyMDNet.tracking.data_prov import RegionExtractor
# from pyMDNet.tracking.run_tracker import *
# from pyMDNet.tracking.bbreg import BBRegressor
# from pyMDNet.tracking.gen_config import gen_config
# import math
# opts = yaml.safe_load(open('../pyMDNet/tracking/options.yaml','r'))
#
# __all__ = ['GlobalATSSTrackVOT_MDNET']
#
#
# class GlobalATSSTrackVOT_MDNET(TrackerVOT):
#
#     # def __init__(self, cfg_file, ckp_file, ckp_file_det, transforms, name_suffix=''):
#     def __init__(self, cfg_file, ckp_file, transforms, name_suffix=''):
#         name = 'GlobalATSSTrackVOT_MDNET'
#         if name_suffix:
#             name += '_' + name_suffix
#         super(GlobalATSSTrackVOT_MDNET, self).__init__(
#             name=name, is_deterministic=True)
#         self.transforms = transforms
#
#         # build config
#         cfg = Config.fromfile(cfg_file)
#         if cfg.get('cudnn_benchmark', False):
#             torch.backends.cudnn.benchmark = True
#         cfg.model.pretrained = None
#         self.cfg = cfg
#
#         # build model
#         model = build_detector(
#             cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
#         fp16_cfg = cfg.get('fp16', None)
#         if fp16_cfg is not None:
#             wrap_fp16_model(model)
#         # 第一遍加载detection模型
#         # checkpoint = load_checkpoint(
#         #     model, ckp_file_det, map_location='cpu')
#         #
#         checkpoint = load_checkpoint(
#             model, ckp_file, map_location='cpu')
#         model.CLASSES = ('object',)
#
#         # GPU usage
#         cuda = torch.cuda.is_available()
#         self.device = torch.device('cuda:0' if cuda else 'cpu')
#         self.model = model.to(self.device)
#
#         self.last_bbox = None
#         self.alpha = 0.3
#
#
#     def init(self, img, bbox):
#         with torch.no_grad():
#             self.model.eval()
#
#             img_init = img
#             # prepare query data
#             img_meta = {'ori_shape': img.shape}
#             bboxes = np.expand_dims(bbox, axis=0)
#             img, img_meta, bboxes = \
#                 self.transforms._process_query(img, img_meta, bboxes)
#             img = img.unsqueeze(0).contiguous().to(
#                 self.device, non_blocking=True)
#             bboxes = bboxes.to(self.device, non_blocking=True)
#
#             # initialize the modulator
#             begin = time.time()
#             self.model._process_query(img, [bboxes])
#             times = time.time() - begin
#
#         # pymdnet
#         self.t_id = 0
#         self.last_bbox = bbox # [xmin ymin xmax ymax]
#         self.last_gt = [bbox[1], bbox[0], bbox[3], bbox[2]]  # [ymin xmin ymax xmax]
#         init_gt1 = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
#         # print(init_gt1)
#         self.init_pymdnet(img_init, init_gt1)  # [x1,y1,w,h]
#         return times
#
#     def update(self, img, gt, **kwargs):
#         with torch.no_grad():
#             self.model.eval()
#
#             img_pre = img
#             # prepare gallary data
#             img_meta = {'ori_shape': img.shape}
#
#             img, img_meta, _ = \
#                 self.transforms._process_gallary(img, img_meta, None)
#
#             img = img.unsqueeze(0).contiguous().to(
#                 self.device, non_blocking=True)
#
#             # get detections
#             results = self.model._process_gallary(
#                 img, [img_meta], rescale=True, **kwargs)[0]  # by lc
#
#         ''' local & global '''
#         self.t_id += 1
#         max_ind = results[:, -1].argmax()
#         global_result, global_score = results[max_ind, :4], results[max_ind, 4]
#
#         center_x, center_y = (self.last_bbox[2]+self.last_bbox[0])/2, (self.last_bbox[3]+self.last_bbox[1])/2
#         w,h = int(self.last_bbox[2]-self.last_bbox[0]), int(self.last_bbox[3]-self.last_bbox[1])
#         search = int(math.sqrt(w*h*16))  # 25
#         x_min, y_min, x_max, y_max = center_x-search/2, center_y-search/2, center_x+search/2, center_y+search/2
#         inds_min = (results[:, 0:2] > np.array([x_min, y_min])).sum(axis=1)
#         inds_max = (results[:, 2:4] < np.array([x_max, y_max])).sum(axis=1)
#         inds = inds_max+inds_min
#         # print(inds_max.shape)
#         # print(inds_min.shape)
#         # print(inds.shape)
#         inds = inds == 4
#         if inds.sum()==0:
#             update = True
#             self.last_bbox = global_result
#             score = global_score
#             print('no local')
#         else:
#             local_results = results[inds, :]
#             local_max_ind = local_results[:, -1].argmax()
#             # topk = 2
#             # indices = results[:, -1].argsort()[::-1][0:topk]
#             # local_results, local_scores = local_results[indices, :4], local_results[indices, 4]
#             local_result, local_score = local_results[local_max_ind, :4], local_results[local_max_ind, 4]
#             local_xywh = np.array([local_result[0],local_result[1],local_result[2]-local_result[0],local_result[3]-local_result[1]])
#             global_xywh = np.array([global_result[0],global_result[1],global_result[2]-global_result[0],global_result[3]-global_result[1]])
#             iou = self._compute_iou(local_xywh, global_xywh)
#             # print(local_xywh)
#             # print(global_xywh)
#             # print(iou)
#             if iou>0.7:
#                 md_score_global_local = self.pymdnet_eval(img_pre, np.array([global_xywh]))[0]
#                 if md_score_global_local>0:
#                     update = True
#                     self.last_bbox = global_result
#                     score = 0.99 #global_score  # global_score
#                     print('global equal local right')
#                 else:
#                     print('global equal local false')
#                     update = False
#                     topk = 3
#                     indices = results[:, -1].argsort()[::-1][0:topk]
#                     global_results, global_scores = results[indices, :4], results[indices, 4]
#                     global_xywhs = np.array([global_results[:,0], global_results[:,1], global_results[:,2] - global_results[:,0],
#                                             global_results[:,3] - global_results[:,1]]).transpose()
#                     # print(global_xywhs)
#                     can_scores = self.pymdnet_eval(img_pre, global_xywhs)
#                     max_id = np.argmax(can_scores)
#                     if can_scores[max_id]>0:
#                         score = 0.99
#                         self.last_bbox = global_results[max_id]
#                     else:
#                         score = 0
#                 # print('global equal local')
#             else:
#                 ''' for mdnet '''
#                 md_score_local = self.pymdnet_eval(img_pre, np.array([local_xywh]))[0]  # x1,y1,w,h
#                 md_score_global = self.pymdnet_eval(img_pre, np.array([global_xywh]))[0]  # x1,y1,w,h
#                 if md_score_local < 0 and md_score_global < 0:
#                     score = 0
#                     update = False
#                     print('no right')
#                 elif md_score_local > 0 and md_score_global > 0:
#                     update = False
#                     # local_ = md_score_local * (1 - self.alpha) + self.alpha * local_score
#                     # global_ = md_score_global * (1 - self.alpha) + self.alpha * global_score
#                     if md_score_local > md_score_global:
#                         score = local_score
#                         self.last_bbox = local_result
#                         print('local compare')
#                     else:
#                         score = global_score
#                         self.last_bbox = global_result
#                         print('global compare')
#                 elif md_score_local > 0 and md_score_global < 0:
#                     update = True
#                     score = local_score #local_score  # 0.99
#                     self.last_bbox = local_result
#                     print('local')
#                 elif md_score_local < 0 and md_score_global > 0:
#                     update = True
#                     score = global_score # global_score  # 0.99
#                     self.last_bbox = global_result
#                     print('global')
#
#         self.last_gt = [self.last_bbox[1], self.last_bbox[0], self.last_bbox[3], self.last_bbox[2]]
#         # candidate_scores = self.pymdnet_eval(img, candidate_bboxes)
#         # max_id = np.argmax(candidate_scores)
#
#         if update:
#             self.collect_samples_pymdnet(img_pre)
#
#         self.pymdnet_long_term_update()
#         # print(score)
#         print(self.last_bbox)
#         return self.last_bbox, score
#
#     # pymdnet
#     def init_pymdnet(self, image, init_bbox):
#         target_bbox = np.array(init_bbox)
#         self.last_result = target_bbox
#         self.pymodel = MDNet('../pyMDNet/models/mdnet_imagenet_vid.pth')
#         if opts['use_gpu']:
#             self.pymodel = self.pymodel.cuda()
#         self.pymodel.set_learnable_params(opts['ft_layers'])
#
#         # Init criterion and optimizer
#         self.criterion = BCELoss()
#         init_optimizer = set_optimizer(self.pymodel, opts['lr_init'], opts['lr_mult'])
#         self.update_optimizer = set_optimizer(self.pymodel, opts['lr_update'], opts['lr_mult'])
#
#         tic = time.time()
#
#         # Draw pos/neg samples
#         # print(image.shape)
#         # print(target_bbox)
#         pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
#             target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])   # [image.shape[1],image.shape[0]]
#         # print(pos_examples)
#
#         neg_examples = np.concatenate([
#             SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
#                 target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
#             SampleGenerator('whole', image.size)(
#                 target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
#         neg_examples = np.random.permutation(neg_examples)
#
#         # Extract pos/neg features
#         pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
#         neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
#         self.feat_dim = pos_feats.size(-1)
#
#         # Initial training
#         train(self.pymodel, self.criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], opts=opts)
#         del init_optimizer, neg_feats
#         torch.cuda.empty_cache()
#
#         # Train bbox regressor
#         bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
#                                          opts['aspect_bbreg'])(
#             target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
#         bbreg_feats = forward_samples(self.pymodel, image, bbreg_examples, opts)
#         self.bbreg = BBRegressor(image.size)
#         self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
#         del bbreg_feats
#         torch.cuda.empty_cache()
#         # Init sample generators
#         self.sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
#         self.pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
#         self.neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])
#
#         # Init pos/neg features for update
#         neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
#         neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
#         self.pos_feats_all = [pos_feats]
#         self.neg_feats_all = [neg_feats]
#
#         spf_total = time.time() - tic
#
#     def pymdnet_eval(self, image, samples):
#         sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)
#         return sample_scores[:, 1][:].cpu().numpy()
#
#     def collect_samples_pymdnet(self, image):
#         self.t_id += 1
#         target_bbox = np.array([self.last_gt[1], self.last_gt[0], self.last_gt[3]-self.last_gt[1], self.last_gt[2]-self.last_gt[0]])
#         pos_examples = self.pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
#         if len(pos_examples) > 0:
#             pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
#             self.pos_feats_all.append(pos_feats)
#         if len(self.pos_feats_all) > opts['n_frames_long']:
#             del self.pos_feats_all[0]
#
#         neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
#         if len(neg_examples) > 0:
#             neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
#             self.neg_feats_all.append(neg_feats)
#         if len(self.neg_feats_all) > opts['n_frames_short']:
#             del self.neg_feats_all[0]
#
#     def pymdnet_short_term_update(self):
#         # Short term update
#         nframes = min(opts['n_frames_short'], len(self.pos_feats_all))
#         pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
#         neg_data = torch.cat(self.neg_feats_all, 0)
#         train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
#               opts=opts)
#
#     def pymdnet_long_term_update(self):
#         if self.t_id % opts['long_interval'] == 0:
#             # Long term update
#             pos_data = torch.cat(self.pos_feats_all, 0)
#             neg_data = torch.cat(self.neg_feats_all, 0)
#             train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
#                   opts=opts)
#
#     def _compute_iou(self, boxA, boxB):  # x1,y1,w,h
#         # determine the (x, y)-coordinates of the intersection rectangle
#         xA = max(boxA[0], boxB[0])
#         yA = max(boxA[1], boxB[1])
#         xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#         yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
#
#         if xA < xB and yA < yB:
#             # compute the area of intersection rectangle
#             interArea = (xB - xA) * (yB - yA)
#             # compute the area of both the prediction and ground-truth
#             # rectangles
#             boxAArea = boxA[2] * boxA[3]
#             boxBArea = boxB[2] * boxB[3]
#             # compute the intersection over union by taking the intersection
#             # area and dividing it by the sum of prediction + ground-truth
#             # areas - the intersection area
#             iou = interArea / float(boxAArea + boxBArea - interArea)
#         else:
#             iou = 0
#
#         assert iou >= 0
#         assert iou <= 1.01
#
#         return iou