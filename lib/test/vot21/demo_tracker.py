import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2

import time
import sys
from lib.test.evaluation import Tracker

class STARK_REF_LT(object):
    """STARK base tracker + STARK refinement"""
    def __init__(self, base_tracker='stark_st', base_param='baseline',
                 ref_tracker='stark_ref', ref_param='baseline', use_new_box=True):
        """use_new_box: whether to use the refined box as the new state"""
        self.use_new_box = use_new_box
        # create base tracker
        tracker_info = Tracker(base_tracker, base_param, "lasot", None)
        base_params = tracker_info.get_parameters()
        base_params.visualization, base_params.debug = False, False
        self.tracker = tracker_info.create_tracker(base_params)
        # create refinement module
        ref_info = Tracker(ref_tracker, ref_param, "lasot", None)
        ref_params = ref_info.get_parameters()
        ref_params.visualization, ref_params.debug = False, False
        self.ref = ref_info.create_tracker(ref_params)

    def initialize(self, img_rgb, box):
        """box: list"""
        # init on the 1st frame
        self.H, self.W, _ = img_rgb.shape
        init_info = {'init_bbox': box}
        # init base tracker
        _ = self.tracker.initialize(img_rgb, init_info)
        # init refinement module
        self.ref.initialize(img_rgb, init_info)

    def track(self, img_rgb):
        # track with the base tracker
        outputs, update_flag = self.tracker.track(img_rgb, return_update_flag=True)
        pred_bbox = outputs['target_bbox']
        conf_score = outputs["conf_score"]
        search_global = outputs["global"]
        if search_global:
            can_boxes = outputs["can_boxes"]
        # refine with the refinement module
        outputs_ref = self.ref.track(img_rgb, pred_bbox, update_flag)
        pred_bbox_ref = outputs_ref["target_bbox"]
        if self.use_new_box:
            self.tracker.state = pred_bbox_ref
        if search_global:
            return pred_bbox_ref, conf_score, search_global,can_boxes
        else:
            return pred_bbox_ref, conf_score, search_global

class p_config(object):
    name = 'stark'
    save_results = True
    use_mask = True
    save_training_data = False
    visualization = True


class VOTLT_Results_Saver(object):
    def __init__(self, save_path, video, t):
        result_path = os.path.join(save_path, 'longterm')
        if not os.path.exists(os.path.join(result_path, video)):
            os.makedirs(os.path.join(result_path, video))
        self.g_region = open(os.path.join(result_path, video, video + '_001.txt'), 'w')
        self.g_region.writelines('1\n')
        self.g_conf = open(os.path.join(result_path, video, video + '_001_confidence.value'), 'w')
        self.g_conf.writelines('\n')
        self.g_time = open(os.path.join(result_path, video, video + '_time.txt'), 'w')
        self.g_time.writelines([str(t)+'\n'])

    def record(self, conf, region, t):
        self.g_conf.writelines(["%f" % conf + '\n'])
        self.g_region.writelines(["%.4f" % float(region[0]) + ',' + "%.4f" % float(
            region[1]) + ',' + "%.4f" % float(region[2]) + ',' + "%.4f" % float(region[3]) + '\n'])
        self.g_time.writelines([str(t)+'\n'])

class Region:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

def get_seq_list(Dataset, mode=None, classes=None, video=None):
    if Dataset == "votlt18":
        data_dir = votlt18_dir
    elif Dataset == 'otb':
        data_dir = otb_dir
    elif Dataset == "votlt19":
        data_dir = '/media/lc/da5a7293-2b4e-42c2-9034-7e975b6e61cc/database/lt2019'
    elif Dataset == "tlp":
        data_dir = tlp_dir
    elif Dataset == "lasot":
        data_dir = os.path.join(lasot_dir, classes)
    elif Dataset == 'demo':
        data_dir = '../demo_sequences'

    sequence_list = os.listdir(data_dir)
    sequence_list.sort()
    sequence_list = [title for title in sequence_list if not title.endswith("txt")]
    if video is not None:
        sequence_list = [video]
    # testing_set_dir = '../utils/testing_set.txt'
    # testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
    if mode == 'test' and Dataset == 'lasot':
        print('test data')
        sequence_list = [vid for vid in sequence_list if vid in testing_set]
    elif mode == 'train' and Dataset == 'lasot':
        print('train data')
        sequence_list = [vid for vid in sequence_list if vid not in testing_set]
    else:
        print("all data")

    return sequence_list, data_dir


def get_groundtruth(Dataset, data_dir, video):
    if Dataset == "votlt" or Dataset == "votlt19" or Dataset == "demo":
        sequence_dir = data_dir + '/' + video + '/color/'
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    elif Dataset == "otb":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth_rect.txt'
    elif Dataset == "lasot":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    elif Dataset == "tlp":
        sequence_dir = data_dir + '/' + video + '/img/'
        gt_dir = data_dir + '/' + video + '/groundtruth_rect.txt'
    try:
        groundtruth = np.loadtxt(gt_dir, delimiter=',')
    except:
        groundtruth = np.loadtxt(gt_dir)
    if Dataset == 'tlp':
        groundtruth = groundtruth[:, 1:5]

    return sequence_dir, groundtruth


def run_seq_list(Dataset, p, sequence_list, data_dir):

    base_save_path = os.path.join('./results', p.name, Dataset)
    if not os.path.exists(base_save_path):
        if p.save_results and not os.path.exists(os.path.join(base_save_path, 'eval_results')):
            os.makedirs(os.path.join(base_save_path, 'eval_results'))
        if p.save_training_data and not os.path.exists(os.path.join(base_save_path, 'train_data')):
            os.makedirs(os.path.join(base_save_path, 'train_data'))

    save_dir = '/media/lc/61854679-da08-4161-9df3-62beb989649f/lc/Desktop/LT-cooperation/vis'

    for seq_id, video in enumerate(sequence_list):
        sequence_dir, groundtruth = get_groundtruth(Dataset, data_dir, video)

        if p.save_training_data:
            result_save_path = os.path.join(base_save_path, 'train_data', video + '.txt')
            if os.path.exists(result_save_path):
                continue
        if p.save_results:
            result_save_path = os.path.join(base_save_path, 'eval_results', video + '.txt')
            if os.path.exists(result_save_path):
                continue
        bb_list = []
        image_list = os.listdir(sequence_dir)
        image_list.sort()
        image_list = [im for im in image_list if im.endswith("jpg") or im.endswith("jpeg")]

        tracker = STARK_REF_LT(base_tracker='stark_st', base_param='baseline', ref_tracker="stark_ref",
                               ref_param="baseline",
                               use_new_box=False)
        imagefile = sequence_dir + image_list[0]
        init_box = [groundtruth[0, 0], groundtruth[0, 1], groundtruth[0, 2], groundtruth[0, 3]]
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        tracker.initialize(image, init_box)

        for imagefile in image_list[1:]:
            imagefile = sequence_dir + imagefile
            image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
            temp = tracker.track(image)
            if len(temp)==4:
                b1, conf, search_global, can_boxes = temp
            else:
                b1, conf, search_global = temp
            x1, y1, w, h = b1
            bb_list.append(b1)
            if True:
                '''Visualization'''
                # original image
                image_ori = image[:, :, ::-1].copy()  # RGB --> BGR
                image_name = imagefile.split('/')[-1]
                # cv2.imwrite(save_path, image_ori)
                # tracker box
                image_b = image_ori.copy()
                cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                              (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
                cv2.putText(image_b, str(conf), (int(b1[0]), int(b1[1])),  # str(pred)
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                image_b_name = image_name.replace('.jpg', '_bbox.jpg')
                save_path = os.path.join(save_dir,video, image_b_name)
                if not os.path.exists(os.path.join(save_dir,video)):
                    os.makedirs(os.path.join(save_dir,video))

                if search_global:
                    print('!')
                    print(can_boxes.shape[0])
                    for i in range(can_boxes.shape[0]):  # xywh
                        cv2.rectangle(image_b, (int(can_boxes[i, 0]), int(can_boxes[i, 1])),
                                      (int(can_boxes[i, 0] + can_boxes[i, 2]), int(can_boxes[i, 1] + can_boxes[i, 3])),
                                      (0, 255, 0), 2)  # 绿色

                        cv2.putText(image_b, str(i), # str(i) + ' ' + str(list_pred[i + 1])
                                        (int(can_boxes[i, 0]), int(can_boxes[i, 1])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imwrite(save_path, image_b)
        np.savetxt('./debug/{}.txt'.format(video),np.array(bb_list))

def eval_tracking(Dataset, p, mode=None, video=None):
    if Dataset == 'lasot':
        classes = os.listdir(lasot_dir)
        classes.sort()
        for c in classes:
            sequence_list, data_dir = get_seq_list(Dataset, mode=mode, classes=c)
            run_seq_list(Dataset, p, sequence_list, data_dir)
    elif Dataset in ['votlt18', 'votlt19', 'tlp', 'otb', 'demo']:
        sequence_list, data_dir = get_seq_list(Dataset, video=video)
        run_seq_list(Dataset, p, sequence_list, data_dir)
    else:
        print('Warning: Unknown dataset.')


# test DiMP_LTMU
p = p_config()
p.save_results = True
eval_tracking('votlt19', p=p, video='ballet')