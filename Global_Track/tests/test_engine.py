# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import _init_paths
from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml
import torch

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.core import wrap_fp16_model

from neuron.data.datasets.config import cfg
from task_evaluation import *
from utils import *
from vis import *
from global_track_atss import *
import neuron.data as data
from neuron.data.datasets.json_dataset import JsonDataset
from six.moves import cPickle as pickle
from time_ import *
import pandas as pd
import math
from PIL import Image
np.random.seed(666)
# create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create a handler
logfile = './log/logger.txt'
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)
# create another handler, control board
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# define handler's output format
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add logger to handlers
logger.addHandler(fh)
logger.addHandler(ch)



def save_object(obj, file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    child_func = test_net
    parent_func = test_net_on_dataset

    return parent_func, child_func

def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = ['fsod_test']
    proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()  # test_on_dataset test_net
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(('fsod_test',))):
                dataset_name, proposal_file = get_inference_dataset(i)   # 'fsod_test', None
                output_dir = 'Outputs/fsod_save_dir/test'   # Outputs/fsod_save_dir/test
                results = parent_func(    # test_net_on_dataset
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing   # True, can be False
                )
                all_results.update(results)  # is a dict, containing evaluation results, not one shot detection results

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()
    if check_expected_results and is_parent:  # don't do what??  but first, we need to get one shot detection results.
        check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        log_copy_paste_friendly_results(all_results)

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDataset(root_dir='/home/lc/Downloads/database', subset='test',cache_dir='../tools/cache')  # test_net maybe
    ''' ceshi '''
    # test_timer = Timer()
    # test_timer.tic()
    # # for simplity ,there is no multi-gpu
    # all_boxes = test_net(
    #     args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
    # )
    # test_timer.toc()
    # logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    ''' ceshi endding '''
    f = open('/media/lc/61854679-da08-4161-9df3-62beb989649f/lc/GlobalTrackATSS/tests/Outputs/fsod_save_dir/test/detections.pkl', 'rb')
    info = pickle.load(f)
    all_boxes = info['all_boxes']
    results = evaluate_all(
        dataset, all_boxes, all_segms=None, all_keyps=None, output_dir=output_dir
    )
    return results


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    full_roidb, dataset, start_ind, end_ind, total_num_images, total_num_cls, support_dict = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range
    )  # full_roidb=30000  dataset=JsonDataset class  0-30000  30000  3001  3000dict  30000=10*5*600

    # build model
    cfg_file = '../configs/qg_atss_r50_shot.py'
    # ckp_file = '../tools/work_dirs/experiment_5_plus/epoch_12.pth'
    # ckp_file = '../tools/work_dirs/experiment_17/epoch_12.pth'   # test
    # ckp_file = '../tools/work_dirs/experiment_27_neg/epoch_12.pth'  # test_17
    ckp_file = '../tools/work_dirs/experiment_shot_31/epoch_12.pth'  # test_17
    transforms = data.BasicPairTransforms(train=False)
    tracker_model = OneShotTrack(
        cfg_file, ckp_file, transforms,
        name_suffix='qg_rcnn_r50_shot')
    model = tracker_model.model

    base_real_index = full_roidb[start_ind]['real_index']
    roidb = full_roidb[start_ind:end_ind]

    index_ls = []
    for item in roidb:
        index_ls.append(item['real_index'])
    num_annotations = len(roidb)  # 30000
    num_images = len(list(set(index_ls)))  # 30000
    num_classes = total_num_cls    # 3001 -1 no background class by lc
    all_boxes = empty_results(num_classes, num_images)  # input:(3001,num_images)
    print('part:', num_images)

    timers = defaultdict(Timer)

    for i, entry in enumerate(roidb):

        box_proposals = None  # is set to None

        # Get support box
        index = entry['index']
        assert len(list(set(entry['gt_classes']))) == 1
        query_cls = list(set(entry['gt_classes']))[0]    # eg :1
        query_img = entry['image']

        all_cls = support_dict[query_cls]['all_cls']  # len=5, not real class eg :[1,2,3,4,5]

        support_way = 5
        support_shot = 1

        support_data_all = []  # all is set to list
        support_box_all = []
        support_cls_ls = []

        for cls_id, cls in enumerate(all_cls):
            begin = cls_id * support_shot
            end = (cls_id + 1) * support_shot
            support_data_all[begin:end] = support_dict[cls]['img']  # a list by lc
            support_box_all[begin:end] = support_dict[cls]['box']
            support_cls_ls.append(cls)        # cls is not real class_number

        save_path = './vis'

        query_img_data = read_image(query_img)
        for j in range(support_way):
            support_img = support_data_all[j].unsqueeze(0).contiguous().to(
                tracker_model.device, non_blocking=True)
            bboxes = support_box_all[j].unsqueeze(0).to(tracker_model.device, non_blocking=True)

            # initialize the modulator
            if timers is None:
                timers = defaultdict(Timer)
            timers['im_detect_bbox'].tic()
            tracker_model.model._process_query(support_img, [bboxes])
            cls_boxes_j = tracker_model.update(img=query_img_data, gt=None)  # im_detect is canceled by lc  default=5
            timers['im_detect_bbox'].toc()
            cls_pad = torch.ones((cls_boxes_j.shape[0],1))*support_cls_ls[j]
            if j==0:
                # cls_boxes_i = torch.cat((torch.from_numpy(cls_boxes_j),cls_pad),dim=1)
                cls_boxes_i = torch.cat((cls_boxes_j,cls_pad),dim=1)
            else:
                cls_boxes_j = torch.cat((cls_boxes_j,cls_pad),dim=1)
                cls_boxes_i = torch.cat((cls_boxes_i,cls_boxes_j),dim=0)

        real_index = entry['real_index'] - base_real_index
        for cls in support_cls_ls:
            extend_support_results(real_index, all_boxes, cls_boxes_i[cls_boxes_i[:,5] == cls][:, :5], cls)

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_annotations - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_detect_bbox'].average_time +
                timers['im_detect_mask'].average_time +
                timers['im_detect_keypoints'].average_time
            )
            misc_time = (
                timers['misc_bbox'].average_time +
                timers['misc_mask'].average_time +
                timers['misc_keypoints'].average_time
            )
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_annotations, det_time, misc_time, eta
                )
            )

        if cfg.VIS:
            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            vis_one_image(
                query_img_data[:, :, ::-1],
                '{:d}_{:s}'.format(i, im_name),
                os.path.join(output_dir, 'vis'),
                cls_boxes_i,
                segms=None,
                keypoints=None,
                thresh=cfg.VIS_TH,
                box_alpha=0.8,
                dataset=dataset,
                show_class=True
            )

        ''' ceshi '''
        # cfg_yaml = yaml.dump(cfg)
        # if ind_range is not None:
        #     det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
        # else:
        #     det_name = 'detections.pkl'
        # det_file = os.path.join(output_dir, det_name)
        # print(det_file)
        # save_object(
        #     dict(
        #         all_boxes=all_boxes,
        #         cfg=cfg_yaml
        #     ), det_file
        # )
        # return all_boxes
    ''' ceshi endding '''

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_boxes=all_boxes,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_boxes

def read_image(filename, color_fmt='RGB'):
    assert color_fmt in Image.MODES
    img = Image.open(filename)
    if not img.mode == color_fmt:
        img = img.convert(color_fmt)
    return np.asarray(img)

def prepare_support(entry):

    # Get support box
    transforms = data.BasicPairTransforms(train=False)
    img_path = entry['image']
    all_box = entry['boxes']

    # choose random box
    all_box_num = all_box.shape[0]
    picked_box_id = np.random.choice(range(all_box_num))  # random.choice(range(all_box_num))
    picked_box = all_box[picked_box_id, :][np.newaxis, :]

    img = read_image(img_path)
    img_meta = {'ori_shape': img.shape}
    img, img_meta, bboxes = \
        transforms._process_query(img, img_meta, picked_box)  # eg: bboxes tensor([[264.5833, 125.0000, 301.0417, 397.9167]])

    img = img.unsqueeze(0).contiguous()
    bboxes = bboxes

    return img, bboxes

def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(root_dir='/home/lc/Downloads/database', subset='test',cache_dir='../tools/cache')

    original_roidb, roidb = dataset.get_roidb(gt=True, test_flag=True)   # perform by lc

    # construct support image crops with bounding box
    support_roidb = []
    cnt = 0
    for item_id, item in enumerate(original_roidb):
        gt_classes = list(set(item['gt_classes']))
        for cls in gt_classes:
            item_new = item.copy()
            #item_new['id'] = item_new['id'] * 1000 + int(cls)
            item_new['target_cls'] = int(cls)
            all_cls = item['gt_classes']
            target_idx = np.where(all_cls == cls)[0] 
            item_new['boxes'] = item['boxes'][target_idx]
            item_new['gt_classes'] = item['gt_classes'][target_idx]
            item_new['index'] = cnt
            item_new['real_index'] = item_id
            cnt += 1
            support_roidb.append(item_new)
    print('support annotation number: ', len(support_roidb))
    roidb_img = []
    roidb_cls = []
    roidb_index = []
    for item_id, item in enumerate(support_roidb):
        roidb_img.append(item['image'])
        roidb_cls.append(item['target_cls'])
        roidb_index.append(item['index'])
        assert item_id == item['index']
    data_dict = {'img_ls': roidb_img, 'cls_ls': roidb_cls, 'index': roidb_index}
    # construct dataframe for picking support images
    support_df = pd.DataFrame.from_dict(data_dict)
    # query image summary for each episode for picking support images
    # 10 query 5 support
    episode_num = 600 #500
    query_way_num = 5
    query_shot_num = 10
    total_num_cls = episode_num * query_way_num + 1  # 3001

    support_way_num = 5
    support_shot_num = 1

    support_dict = {}
    for ep in range(episode_num):
        query_img = []
        query_cls = []
        query_index = []
        used_img_ls = []
        query_real_cls = []
        for way in range(query_way_num):
            for shot in range(query_shot_num):
                roidb_id = ep * 50 + way * 10 + shot  # 599*50+4*10+9
                current_roidb = roidb[roidb_id]
                query_img.append(current_roidb['image'])
                query_index.append(current_roidb['index'])
                used_img_ls.append(current_roidb['image'])
            real_cls = current_roidb['target_cls']
            query_real_cls.append(real_cls)
            query_cls.append(list(set(current_roidb['gt_classes']))[0])
        assert len(query_cls) == len(query_real_cls) == query_way_num

        for cls_id, cls in enumerate(query_cls):
            support_dict[cls] = {}
            support_dict[cls]['all_cls'] = query_cls
            support_real_cls = query_real_cls[cls_id]
            for shot in range(support_shot_num):  # only support support_shot_num=1 by lc
                # random_id = ep * 25 + cls_id * 5 + shot
                random_id = ep * 5 + cls_id * 5 + shot  # maybe this random seed is better ? by lc
                support_index = support_df.loc[(support_df['cls_ls'] == support_real_cls) & (~support_df['img_ls'].isin(used_img_ls)), 'index'].sample(random_state=random_id).tolist()[0]
                current_support = support_roidb[support_index]
                img_name = current_support['image']
                # support_img, support_box = crop_support(current_support)  # original format
                support_img, support_box = prepare_support(current_support)
                support_dict[cls]['img'] = support_img
                support_dict[cls]['box'] = support_box   # if shot>1, please concat, no pre-difining  np not cuda, because of memory burden
                used_img_ls.append(img_name)

    if ind_range is not None:
        total_num_images = len(roidb) # it is the query roidb
        start, end = ind_range
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images, total_num_cls, support_dict


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]  # same as co-attention's paper coding  by lc

    return all_boxes

def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]

def extend_support_results(index, all_res, im_res, cls_idx):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    #for cls_idx in range(1, len(im_res)):
    all_res[cls_idx][index] = im_res #[1]
