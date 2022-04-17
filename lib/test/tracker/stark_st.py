import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
from copy import deepcopy
# for debug
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.stark import build_starkst
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box

''' added '''
from lib.utils.multitracker import AsscoiateTracker

import numpy as np
# Global Tracker
import Global_Track._init_paths
import neuron.data as data
from Global_tracker import *
import collections
import cv2
import matplotlib.pylab as plt
import scipy

from vot_path import base_path
import sys
sys.path.append(base_path + '/pyMDNet/modules')
sys.path.append(base_path + '/pyMDNet/tracking')
print(sys.path)
# pymdnet
from pyMDNet.modules.model import *
sys.path.insert(0, base_path + '/pyMDNet')
from pyMDNet.modules.model import MDNet, BCELoss, set_optimizer
from pyMDNet.modules.sample_generator import SampleGenerator
from pyMDNet.modules.utils import overlap_ratio
from pyMDNet.tracking.data_prov import RegionExtractor
from pyMDNet.tracking.run_tracker import *
from bbreg import BBRegressor
from gen_config import gen_config
opts = yaml.safe_load(open(base_path + '/pyMDNet/tracking/options.yaml','r'))


class STARK_ST(BaseTracker):
    def __init__(self, params, dataset_name):
        super(STARK_ST, self).__init__(params)
        network = build_starkst(params.cfg, pretrained=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # template update
        self.z_dict1 = {}
        self.z_dict_list = []
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)
        self.num_extra_template = len(self.update_intervals)

    def initialize(self, image, info: dict):
        # initialize z_dict_list
        self.z_dict_list = []
        # get the 1st template
        z_patch_arr1, _, z_amask_arr1 = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                      output_sz=self.params.template_size)
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1)
        # get the complete z_dict_list
        self.z_dict_list.append(self.z_dict1)
        for i in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

        ''' added for redet  '''
        #  [------------------------starting---------------------------]
        state = info['init_bbox']
        init_gt = [state[0], state[1], state[0] + state[2], state[1] + state[3]]  # xmin,ymin,xmax,ymax
        self.init_pymdnet(image, state)  # x1y1wh
        self.last_bbox = state
        self.Golbal_Track_init(image, init_gt)
        self.search_global = False
        self.re = 0
        self.last_flag_global = False
        self.cnt_redet = 0
        self.spatio_start = False
        self.first_redet = True
        self.cnt_redet_flag = False
        self.t_id = 0
        self.last_state = state
        #  [------------------------ending---------------------------]

    #  [------------------------starting---------------------------]
    def Golbal_Track_init(self, image, init_box):
        cfg_file = base_path+'/Global_Track/configs/qg_atss_r50_joint2.py'
        ckp_file = base_path+'/Global_Track/checkpoints/12_13_multi.pth'
        transforms = data.BasicPairTransforms(train=False)
        self.Global_Tracker = GlobalATSSTrack(
            cfg_file, ckp_file, transforms,
            name_suffix='12_13_multi')
        self.Global_Tracker.init(image, init_box)

    # -------mdnet-------
    # ------------------------------------------------------------------------------------------------------------------
    def init_pymdnet(self, image, init_bbox):
        target_bbox = np.array(init_bbox)
        self.last_result = target_bbox
        self.pymodel = MDNet(
            base_path+'/pyMDNet/models/mdnet_imagenet_vid.pth')
        if opts['use_gpu']:
            self.pymodel = self.pymodel.cuda()
        self.pymodel.set_learnable_params(opts['ft_layers'])

        # Init criterion and optimizer
        self.criterion = BCELoss()
        init_optimizer = set_optimizer(self.pymodel, opts['lr_init'], opts['lr_mult'])
        self.update_optimizer = set_optimizer(self.pymodel, opts['lr_update'], opts['lr_mult'])

        tic = time.time()

        # Draw pos/neg samples
        pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
            target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        neg_examples = np.concatenate([
            SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
            SampleGenerator('whole', image.size)(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.feat_dim = pos_feats.size(-1)

        # Initial training
        train(self.pymodel, self.criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], opts=opts)
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                         opts['aspect_bbreg'])(
            target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
        bbreg_feats = forward_samples(self.pymodel, image, bbreg_examples, opts)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()
        # Init sample generators
        self.sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
        self.pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

        # Init pos/neg features for update
        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]

    def collect_samples_pymdnet(self, image):
        self.t_id += 1
        target_bbox = np.array(
            [self.last_bbox[0], self.last_bbox[1], self.last_bbox[2], self.last_bbox[3]])  # xmin,ymin,w,h
        pos_examples = self.pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
        if len(pos_examples) > 0:
            pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
            self.pos_feats_all.append(pos_feats)
        if len(self.pos_feats_all) > opts['n_frames_long']:
            del self.pos_feats_all[0]

        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
        if len(neg_examples) > 0:
            neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
            self.neg_feats_all.append(neg_feats)
        if len(self.neg_feats_all) > opts['n_frames_short']:
            del self.neg_feats_all[0]

    def pymdnet_short_term_update(self):
        # Short term update
        nframes = min(opts['n_frames_short'], len(self.pos_feats_all))
        pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
        neg_data = torch.cat(self.neg_feats_all, 0)
        train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
              opts=opts)

    def pymdnet_long_term_update(self):
        if self.t_id % opts['long_interval'] == 0:
            # Long term update
            pos_data = torch.cat(self.pos_feats_all, 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
                  opts=opts)

    def pymdnet_eval(self, image, samples):
        sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)
        return sample_scores[:, 1][:].cpu().numpy()

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
    # ------------------------------------------------------------------------------------------------------------------

    #  [------------------------ending---------------------------]

    def track(self, image, info: dict = None, return_update_flag=False):
        update_flag = False
        H, W, _ = image.shape
        # ------starting-------
        self.img_w = W
        self.img_h = H
        self.last_image = image
        # ------ending-------
        self.frame_id += 1

        # if not self.search_global:
        # get the t-th search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = self.z_dict_list + [x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)  # [x1,y1,w,h]
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()

        ''' added for ratio '''
        if not self.last_flag_global:
            ratio = self.state[2]*self.state[3]/(self.last_state[2]*self.last_state[3])
            if ratio>2 or ratio<0.6:
                conf_score = 0.0
        self.last_state = self.state

        ''' added for redet  '''
        # if self.state[
        if conf_score<0.10:  # 0.20
            if not self.search_global:
                self.search_global = True
                flag = 'not_found'
                self.count = 0  #相当于要保持base tracker和mdnet的同时满足要求
        else:
            flag = 'normal'

        if self.cnt_redet_flag == True:
            self.cnt_redet += 1

        #  [------------------------starting---------------------------]
        # [---------------- Global re-detection session  ----------------]
        if self.search_global:
            self.re += 1
            if self.cnt_redet >= 30 or self.first_redet:
                spatio_flag = True
                self.cnt_redet = 0
            else:
                spatio_flag = False
            self.cnt_redet_flag = False
            if self.first_redet == True:
                self.first_redet = False
            print('re-detection'+str(self.re)+str(self.frame_id))
            self.global_num = 10
            # find candidate
            # def Global_Track_eval(self, image, num):  # xywh
            results = self.Global_Tracker.update(image, self.frame_id, threshold=0.1)
            for i in range(results.shape[0]):
                results[i, 0] = max(results[i, 0], 0)
                results[i, 1] = max(results[i, 1], 0)
                results[i, 2] = min(results[i, 2], image.shape[1])
                results[i, 3] = min(results[i, 3], image.shape[0])

            index = np.argsort(results[:, -1])[::-1]
            # if len(index)>self.global_num:
            #     max_index = index[:self.global_num]
            # else:
            #     max_index = index
            max_index = index

            can_boxes = results[max_index][:, :4]
            can_boxes_wcon = results[max_index]  # 有confidence的
            # print(can_boxes)
            can_boxes_confidences = results[max_index][:, 4]
            for i in range(can_boxes.shape[0]):
                if can_boxes[i, 0] < 0:
                    can_boxes[i, 0] = 0
                if can_boxes[i, 1] < 0:
                    can_boxes[i, 1] = 0
                if can_boxes[i, 2] >= self.img_w:
                    can_boxes[i, 2] = self.img_w - 1
                if can_boxes[i, 3] >= self.img_h:
                    can_boxes[i, 3] = self.img_h - 1
            can_boxes_x1y1x2y2 = can_boxes
            can_boxes = np.array([can_boxes[:, 0], can_boxes[:, 1], can_boxes[:, 2] - can_boxes[:, 0],
                                  can_boxes[:, 3] - can_boxes[:, 1]]).transpose()  # xywh

            can_boxes_center = np.stack(
                (can_boxes[:, 1] + (can_boxes[:, 3] - 1) / 2, can_boxes[:, 0] + (can_boxes[:, 2] - 1) / 2),
                axis=1)  # 有，
            can_boxes_hw = np.stack((can_boxes[:, 3], can_boxes[:, 2]), axis=1)

            can_score = results[max_index][:, 4]

            pos = torch.Tensor([self.state[1]+self.state[3]/2,self.state[0]+self.state[2]/2])
            target_sz = torch.Tensor([self.state[3],self.state[2]])
            # list_search_pos = [pos]
            # list_search_target_sz = [target_sz]
            list_search_pos = []
            list_search_target_sz = []
            list_can_score = []
            for s in can_score:
                list_can_score.append(s)

            # ---------------- adding redetection results --------------
            # if self.params.get('temporal_using', False):
            if True:
                if not self.last_flag_global and spatio_flag:
                    self.spatio_start = True

                    self.distractor_track = []  # 每次重新进入全局模式都重新建立，bird1这种频繁进入的暂且不管
                    self.last_global_num = 10  # self.params.get('global_num', 10)
                    # 准备上一帧的结果作为初始化
                    last_global_results = self.Global_Tracker.update(self.last_image, self.frame_id - 1,
                                                                     threshold=0.1)  # threshold=0.01
                    # 初始化一个联系跟踪器
                    self.associate = AsscoiateTracker()
                    # 去掉和上一帧的预测框iou>0的
                    last_global_iou_ = self.compute_iou(
                        torch.cat((self.last_bbox[[0, 1]], self.last_bbox[[0, 1]] + self.last_bbox[[2, 3]])),
                        last_global_results)  # (x1y1x2y2)
                    mid_index = (last_global_iou_ == 0)
                    last_global_dets = last_global_results[mid_index.numpy(),
                                       :]  # last_global_results[mid_index] false!when==1
                    last_global_index = np.argsort(last_global_dets[:, -1])[::-1]
                    if len(last_global_index) > self.last_global_num:
                        last_global_max_index = last_global_index[:self.last_global_num]
                    else:
                        last_global_max_index = last_global_index
                    last_global_dets = last_global_dets[last_global_max_index,
                                       :]  # (10,5) # last_global_results[mid_index] false!when==1

                    self.associate.initialize(last_global_dets)
                    output_stracks, u_detection,_ = self.associate.update(can_boxes_wcon)
                    # m_detection = []
                    # for i in range(can_boxes_wcon.shape[0]):
                    #     if i in u_detection:
                    #         m_detection.append(i)
                    # match_detection = can_boxes_wcon[m_detection, :]
                    can_boxes_wcon = can_boxes_wcon[u_detection, :]
                    # delete <0.1
                    can_boxes_center = can_boxes_center[u_detection, :]
                    can_boxes_hw = can_boxes_hw[u_detection, :]
                    can_boxes = can_boxes[u_detection, :]
                    can_boxes_center = can_boxes_center[can_boxes_wcon[:, -1] > 0.1, :]
                    can_boxes_hw = can_boxes_hw[can_boxes_wcon[:, -1] > 0.1, :]
                    can_boxes = can_boxes[can_boxes_wcon[:, -1] > 0.1, :]
                    can_boxes_wcon = can_boxes_wcon[can_boxes_wcon[:, -1] > 0.1, :]
                    # -------------------------------------------------------
                    index = np.argsort(can_boxes_wcon[:, -1])[::-1]
                    if len(index) > self.global_num:
                        max_index = index[:self.global_num]
                    else:
                        max_index = index
                    can_boxes_center = can_boxes_center[max_index][:, :4]
                    can_boxes_hw = can_boxes_hw[max_index][:, :4]
                    can_boxes = can_boxes[max_index][:, :4]
                    can_boxes_wcon = can_boxes_wcon[max_index]
                    # -------------------------------------------------------


                elif self.last_flag_global and self.spatio_start:  # 上一帧已经是全图检测了
                    if [track.track_id for track in self.associate.tracked_stracks] == [] and [track.track_id for track
                                                                                               in
                                                                                               self.associate.lost_stracks] == []:
                        # pass  # if no tracks is maintained, don't perform
                        can_boxes_center = can_boxes_center[can_boxes_wcon[:, -1] > 0.1, :]
                        can_boxes_hw = can_boxes_hw[can_boxes_wcon[:, -1] > 0.1, :]
                        can_boxes = can_boxes[can_boxes_wcon[:, -1] > 0.1, :]  # 20210321
                        can_boxes_wcon = can_boxes_wcon[can_boxes_wcon[:, -1] > 0.1, :]  # 20210321
                        # print(self.frame_num, can_boxes.shape[0])
                        # -------------------------------------------------------
                        index = np.argsort(can_boxes_wcon[:, -1])[::-1]
                        if len(index) > self.global_num:
                            max_index = index[:self.global_num]
                        else:
                            max_index = index
                        can_boxes_center = can_boxes_center[max_index][:, :2]
                        can_boxes_hw = can_boxes_hw[max_index][:, :2]
                        can_boxes = can_boxes[max_index][:, :4]
                        can_boxes_wcon = can_boxes_wcon[max_index]
                        # -------------------------------------------------------

                    else:
                        output_stracks, u_detection,_ = self.associate.update(can_boxes_wcon)
                        # current frame matched detections can't be the target
                        # m_detection = []
                        # for i in range(can_boxes_wcon.shape[0]):
                        #     if i in u_detection:
                        #         m_detection.append(i)
                        # match_detection = can_boxes_wcon[m_detection, :]
                        can_boxes_wcon = can_boxes_wcon[u_detection, :]
                        # delete <0.1
                        can_boxes_center = can_boxes_center[u_detection, :]
                        can_boxes_hw = can_boxes_hw[u_detection, :]
                        can_boxes = can_boxes[u_detection, :]
                        can_boxes_center = can_boxes_center[can_boxes_wcon[:, -1] > 0.1, :]
                        can_boxes_hw = can_boxes_hw[can_boxes_wcon[:, -1] > 0.1, :]
                        can_boxes = can_boxes[can_boxes_wcon[:, -1] > 0.1, :]
                        can_boxes_wcon = can_boxes_wcon[can_boxes_wcon[:, -1] > 0.1, :]

                        for j in range(len(self.associate.tracked_stracks)):
                            tracklet = self.associate.tracked_stracks[j].tlbr
                            reserve_ = []
                            for s in range(can_boxes_wcon.shape[0]):
                                if can_boxes_wcon[s, 0] > tracklet[0] and can_boxes_wcon[s, 1] > tracklet[1] and \
                                        can_boxes_wcon[s, 2] < tracklet[2] and can_boxes_wcon[s, 3] < tracklet[3]:
                                    # pass
                                    self.associate.initialize(can_boxes_wcon[s:s + 1, :], add=True)
                                else:
                                    reserve_.append(s)
                            can_boxes_wcon = can_boxes_wcon[reserve_, :]
                            can_boxes_center = can_boxes_center[reserve_, :]
                            can_boxes_hw = can_boxes_hw[reserve_, :]
                            can_boxes = can_boxes[reserve_, :]
                        # -------------------------------------------------------
                        index = np.argsort(can_boxes_wcon[:, -1])[::-1]
                        if len(index) > self.global_num:
                            max_index = index[:self.global_num]
                        else:
                            max_index = index
                        can_boxes_center = can_boxes_center[max_index][:, :2]
                        can_boxes_hw = can_boxes_hw[max_index][:, :2]
                        can_boxes = can_boxes[max_index][:, :4]
                        can_boxes_wcon = can_boxes_wcon[max_index]
                        # -------------------------------------------------------
                else:
                    index = np.argsort(can_boxes_wcon[:, -1])[::-1]
                    if len(index) > self.global_num:
                        max_index = index[:self.global_num]
                    else:
                        max_index = index
                    can_boxes_center = can_boxes_center[max_index][:, :4]
                    can_boxes_hw = can_boxes_hw[max_index][:, :4]
                    can_boxes = can_boxes[max_index][:, :4]
                    can_boxes_wcon = can_boxes_wcon[max_index]
            # ___________________________________________________________________________________________________________________________
            for can in can_boxes_center:
                list_search_pos.append(torch.from_numpy(can))
            for can_hw in can_boxes_hw:
                list_search_target_sz.append(torch.from_numpy(can_hw))

            list_pred = []
            # for i in range(len(list_search_pos)):
            #     pred = self.pymdnet_eval(image, torch.cat((list_search_pos[i][[1, 0]] - (
            #             list_search_target_sz[i][[1, 0]] - 1) / 2, list_search_target_sz[i][[1, 0]])).unsqueeze(0))
            #     list_pred.append(pred.item())

            for i in range(len(list_search_pos)):
                pos = list_search_pos[i]
                target_sz = list_search_target_sz[i]
                state = [pos[1].item() - target_sz[1].item() / 2, pos[0].item() - target_sz[0].item() / 2,
                         target_sz[1].item(), target_sz[0].item()]
                x_patch_arr, resize_factor, x_amask_arr = sample_target(image, state, 2.0,
                                                                        output_sz=self.params.search_size)  # (x1, y1, w, h)
                search = self.preprocessor.process(x_patch_arr, x_amask_arr)
                with torch.no_grad():
                    x_dict = self.network.forward_backbone(search)
                    # merge the template and the search
                    feat_dict_list = self.z_dict_list + [x_dict]
                    seq_dict = merge_template_search(feat_dict_list)
                    # run the transformer
                    out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True,
                                                                      run_cls_head=True)
                # get the final result
                pred_boxes = out_dict['pred_boxes'].view(-1, 4)
                # Baseline: Take the mean of all pred boxes as the final result
                pred_box = (pred_boxes.mean(
                    dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                # get the final box result
                state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
                # get confidence score (whether the search region is reliable)
                conf_score_global = out_dict["pred_logits"].view(-1).sigmoid().item()

                list_pred.append(conf_score_global)

            if len(list_search_pos) == 0:
                flag = 'not_found'
            else:
                find_i = list_pred.index(max(list_pred))  # 现在判断pred_meta
                pos = list_search_pos[find_i]
                target_sz = list_search_target_sz[find_i]
                print(list_pred[find_i])
                # if list_pred[find_i] > 0:
                #     flag = 'normal'
                # else:
                #     flag = 'not_found'
                if list_pred[find_i] > 0.25:
                    flag = 'normal'
                else:
                    flag = 'not_found'

            self.last_flag_global = self.search_global  # settings

            # ----------------------------------------------------------------
            # ---------------- [re-detection flag] ----------------#
            if flag == 'not_found':
                self.search_global = True
                self.count = 0
                conf_score = 0

            else:
                # state = [pos[1].item() - target_sz[1].item() / 2, pos[0].item() - target_sz[0].item() / 2,
                #          target_sz[1].item(), target_sz[0].item()]
                # x_patch_arr, resize_factor, x_amask_arr = sample_target(image, state, self.params.search_factor,
                #                                                         output_sz=self.params.search_size)  # (x1, y1, w, h)
                # search = self.preprocessor.process(x_patch_arr, x_amask_arr)
                # with torch.no_grad():
                #     x_dict = self.network.forward_backbone(search)
                #     # merge the template and the search
                #     feat_dict_list = self.z_dict_list + [x_dict]
                #     seq_dict = merge_template_search(feat_dict_list)
                #     # run the transformer
                #     out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True,
                #                                                       run_cls_head=True)
                # # get the final result
                # pred_boxes = out_dict['pred_boxes'].view(-1, 4)
                # # Baseline: Take the mean of all pred boxes as the final result
                # pred_box = (pred_boxes.mean(
                #     dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                # # get the final box result
                # state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
                # # get confidence score (whether the search region is reliable)
                # conf_score_global = out_dict["pred_logits"].view(-1).sigmoid().item()

                # if conf_score_global>0.25:
                #     if self.last_flag == 'not_found':
                #         self.count = 1  # self.target_sz = target_sz
                #         # conf_score = 0.5
                #     else:
                #         self.count += 1  # self.target_sz = target_sz
                #         # conf_score = 0.5
                #
                #     if self.count == 5:
                #         self.search_global = False
                #         self.count = 0
                #         conf_score = 1.0
                #         self.state = [pos[1].item() - target_sz[1].item() / 2, pos[0].item() - target_sz[0].item() / 2,
                #                       target_sz[1].item(), target_sz[0].item()]
                #         # ---------------展开重检测后的redet，固定30以上才能用distractor更新
                #         self.cnt_redet = 0
                #         self.cnt_redet_flag = True
                #         self.last_flag_global = self.search_global  # settings  这个量必须要纠正回来，不然会影响第二次及以后的重检测部分  ！！！！！！！！
                #         self.spatio_start = False
                #         # --------------------------------------------------------------------------------------------------
                #     else:
                #         self.search_global = True
                # else:
                #     self.search_global = True


                if self.last_flag == 'not_found':
                    self.count = 1  # self.target_sz = target_sz
                    # conf_score = 0.5
                else:
                    self.count += 1  # self.target_sz = target_sz
                    # conf_score = 0.5

                if self.count == 5:
                    self.search_global = False
                    self.count = 0
                    conf_score = 1.0
                    self.state = [pos[1].item() - target_sz[1].item() / 2, pos[0].item() - target_sz[0].item() / 2,
                                  target_sz[1].item(), target_sz[0].item()]
                    # ---------------展开重检测后的redet，固定30以上才能用distractor更新
                    self.cnt_redet = 0
                    self.cnt_redet_flag = True
                    self.last_flag_global = self.search_global  # settings  这个量必须要纠正回来，不然会影响第二次及以后的重检测部分  ！！！！！！！！
                    self.spatio_start = False
                    # --------------------------------------------------------------------------------------------------
                else:
                    self.search_global = True


        #
        #     # ---------------- Global re-detection session endding ----------------

        self.last_flag_global = self.search_global
        self.last_flag = flag  # settings
        self.last_bbox = torch.Tensor(self.state)
        # #  [------------------------ending---------------------------]

        # update template
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > 0.5 and self.search_global==False:
                update_flag = True
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                self.z_dict_list[idx+1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame
                # ------- UPDATE MDNet module ------- #
        # if update_flag:
        #     self.collect_samples_pymdnet(image)
        # self.pymdnet_long_term_update()
                # --------------------------------------------------------------------------------------------------------------

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score}
        else:
            if return_update_flag:
                if self.search_global:
                    return {"target_bbox": self.state,
                        "conf_score": conf_score,"global": self.search_global,"can_boxes": can_boxes}, update_flag
                else:
                    return {"target_bbox": self.state,
                            "conf_score": conf_score, "global": self.search_global}, update_flag
            else:
                return {"target_bbox": self.state,
                        "conf_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return STARK_ST
