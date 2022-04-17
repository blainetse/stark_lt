import numpy as np
from time_ import *
from collections import defaultdict

def im_detect_all(model, im, support_data, support_box, support_cls, support_shot, query_path, box_proposals=None,
                  timers=None):
    """Process the outputs of model for testing
    Args:
      model: the network module
      im_data: Pytorch variable. Input batch to the model.
      im_info: Pytorch variable. Input batch to the model.
      gt_boxes: Pytorch variable. Input batch to the model.
      num_boxes: Pytorch variable. Input batch to the model.
      args: arguments from command line.
      timer: record the cost of time for different steps
    The rest of inputs are of type pytorch Variables and either input to or output from the model.
    """
    if timers is None:
        timers = defaultdict(Timer)
    timers['im_detect_bbox'].tic()

    # perform this condition, by lc
    scores, boxes, im_scale, blob_conv = im_detect_bbox(
        model, im, support_data, support_box, support_cls, query_path, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE,
        box_proposals)

    all_cls = scores[:, -1]
    all_scores = scores[:, :-1]
    all_cls = all_cls[:, np.newaxis]
    timers['im_detect_bbox'].toc()
    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class) (numpy.ndarray)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()

    for cls_id, cls in enumerate(support_cls):
        cls_inds = np.where(all_cls == cls)[0]
        scores_now, boxes_now, cls_boxes_now, nms_cls = box_results_with_nms_and_limit(all_scores[cls_inds],
                                                                                       boxes[cls_inds],
                                                                                       all_cls[cls_inds])
        if cls_id == 0:
            cls_boxes = cls_boxes_now
        else:
            cls_boxes[1] = np.concatenate((cls_boxes[1], cls_boxes_now[1]), axis=0)

    timers['misc_bbox'].toc()

    # no seg and no keypoints
    cls_segms = None
    cls_keyps = None

    return cls_boxes, cls_segms, cls_keyps