# for fsod-test show, 1-way 1-shot
import _init_paths
import torch
import cv2
import neuron.data as data
from neuron.data.datasets.json_dataset import JsonDataset
from neuron.data.datasets.config import cfg
from global_track_atss import *
import numpy as np
from PIL import Image

def rank_for_training(roidb):
    """Rank the roidb entries according to image aspect ration and mark for cropping
    for efficient batching if image is too long.

    Returns:
        ratio_list: ndarray, list of aspect ratios from small to large
        ratio_index: ndarray, list of roidb entry indices correspond to the ratios
    """
    RATIO_HI = cfg.TRAIN.ASPECT_HI  # largest ratio to preserve.
    RATIO_LO = cfg.TRAIN.ASPECT_LO  # smallest ratio to preserve.

    need_crop_cnt = 0

    ratio_list = []
    cls_list = []
    id_list = []
    for entry in roidb:
        width = entry['width']
        height = entry['height']
        ratio = width / float(height)
        target_cls = entry['target_cls']
        img_id = entry['id'] #int(str(entry['id'])[:-3])

        if cfg.TRAIN.ASPECT_CROPPING:
            if ratio > RATIO_HI:
                entry['need_crop'] = True
                ratio = RATIO_HI
                need_crop_cnt += 1
            elif ratio < RATIO_LO:
                entry['need_crop'] = True
                ratio = RATIO_LO
                need_crop_cnt += 1
            else:
                entry['need_crop'] = False
        else:
            entry['need_crop'] = False

        ratio_list.append(ratio)
        cls_list.append(target_cls)
        id_list.append(img_id)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    cls_list = np.array(cls_list)
    id_list = np.array(id_list)
    return ratio_list[ratio_index], ratio_index, cls_list, id_list

cfg_file = '../configs/qg_atss_r50_shot.py'
ckp_file = '../tools/work_dirs/experiment_27_neg/epoch_3.pth'
# ckp_file = '../tools/work_dirs/experiment_21_neg/epoch_2.pth'
# ckp_file = '../tools/work_dirs/experiment_24_midloss/epoch_3.pth'
transforms = data.BasicPairTransforms(train=False)
tracker_model = OneShotTrack(
    cfg_file, ckp_file, transforms,
    name_suffix='qg_atss_cbam_tracking')

def show_image_shot(img, bboxes=None, score=None, attentions=None, bbox_fmt='ltrb', colors=None,
               thickness=3, fig=1, delay=0, max_size=640,
               visualize=True, cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if bboxes is not None:
            bboxes = np.array(bboxes, dtype=np.float32) * scale

    if bboxes is not None:
        assert bbox_fmt in ['ltwh', 'ltrb']
        bboxes = np.array(bboxes, dtype=np.int32)
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
        if bboxes.shape[1] == 4 and bbox_fmt == 'ltwh':
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:] - 1

        # clip bounding boxes
        h, w = img.shape[:2]
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, bbox in enumerate(bboxes):
            color = colors[i % len(colors)]
            if len(bbox) == 4:
                pt1 = (int(bbox[0]), int(bbox[1]))
                pt2 = (int(bbox[2]), int(bbox[3]))
                img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
                img = cv2.putText(img, str(score[i]),
                                  (bbox[0], bbox[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # by lc
            else:
                pts = bbox.reshape(-1, 2)
                img = cv2.polylines(img, [pts], True, color.tolist(), thickness)

    if visualize:
        if isinstance(fig, str):
            winname = fig
        else:
            winname = 'window_{}'.format(fig)

        if attentions is not None:
            h, w = img.shape[:2]
            attention_list = [cv2.resize(o,(h,w)) for o in attentions]
            attention = np.zeros((h,w))
            for o in attention_list:
                attention += o
            attention /= len(attention_list)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    if cvt_code in [cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2RGB]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def read_image(filename, color_fmt='RGB'):
    assert color_fmt in Image.MODES
    img = Image.open(filename)
    if not img.mode == color_fmt:
        img = img.convert(color_fmt)
    return np.asarray(img)

dataset = JsonDataset(root_dir='/home/lc/Downloads/database', subset='test',cache_dir='../tools/cache')
cache = dataset.id_to_category_map
original_roidb = dataset.get_roidb(gt=True, test_flag=False)  # False only means that we don't do episodes-preparing
# new dataset split according to class
roidb = []
for item in original_roidb:
    gt_classes = list(set(item['gt_classes']))
    all_cls = np.array(item['gt_classes'])

    for cls in gt_classes:
        item_new = item.copy()
        target_idx = np.where(all_cls == cls)[0]
        #item_new['id'] = item_new['id'] * 1000 + int(cls)
        item_new['target_cls'] = int(cls)
        item_new['boxes'] = item_new['boxes'][target_idx]
        item_new['max_classes'] = item_new['max_classes'][target_idx]
        item_new['gt_classes'] = item_new['gt_classes'][target_idx]
        item_new['is_crowd'] = item_new['is_crowd'][target_idx]
        item_new['segms'] = item_new['segms'][:target_idx.shape[0]]
        item_new['seg_areas'] = item_new['seg_areas'][target_idx]
        item_new['max_overlaps'] = item_new['max_overlaps'][target_idx]
        item_new['box_to_gt_ind_map'] = np.array(range(item_new['gt_classes'].shape[0]))
        item_new['gt_overlaps'] = item_new['gt_overlaps'][target_idx]
        roidb.append(item_new)

ratio_list, ratio_index, cls_list, id_list = rank_for_training(roidb)
info_list = np.concatenate([ratio_index[:, np.newaxis], cls_list[:, np.newaxis]], axis=1)
info_list = np.concatenate([info_list, id_list[:, np.newaxis]], axis=1)
dataset_ = data.RoiDataLoader(
            roidb,
            201,  # 801 model.classes, not used
            info_list,
            ratio_list,
            training=False)

ranid = np.random.choice(range(len(roidb))) +1 ##########

query_roidb = roidb[ranid]
query_img = read_image(query_roidb['image'])
query_cls = query_roidb['target_cls']
support_index = dataset_.index_pd.loc[
            (dataset_.index_pd['cls_ls'] == query_cls) , 'index'].sample(random_state=ranid).tolist()[0]+50   ##########
support_roidb = roidb[support_index]
support_img = read_image(support_roidb['image'])
support_boxes = support_roidb['boxes']
all_box_num = support_boxes.shape[0]
picked_box_id = np.random.choice(range(all_box_num))
picked_box = support_boxes[picked_box_id, :][np.newaxis, :]

img_meta = {'ori_shape': support_img.shape}
support_img, img_meta, bboxes = \
     transforms._process_query(support_img, img_meta, picked_box)
support_img = support_img.unsqueeze(0).contiguous().to(tracker_model.device, non_blocking=True)
bboxes = bboxes.to(tracker_model.device, non_blocking=True)

# initialize the modulator
tracker_model.model._process_query(support_img, [bboxes])
cls_boxes_j = tracker_model.update(img=query_img, gt=None)
print(query_cls,cache[query_cls])
print(support_roidb['target_cls'], cache[support_roidb['target_cls']])
print(query_roidb['boxes'])
print(cls_boxes_j)
print(query_img.shape)
show_image_shot(query_img, cls_boxes_j[:,:4], cls_boxes_j[:,4])
print(query_cls)
