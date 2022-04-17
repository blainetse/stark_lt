import lap
import numpy as np
import scipy
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from .kalman_filter import *


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def giou(atlbrs, btlbrs):
    '''
    cal GIOU of two boxes or batch boxes
    :type atlbrs: list[tlbr] | np.ndarray (n, 4)
    :type atlbrs: list[tlbr] | np.ndarray (k, 4)

    :rtype ious np.ndarray (n, k)
    '''
    if isinstance(atlbrs,list):
        atlbrs = np.array(atlbrs)
        btlbrs = np.array(btlbrs)
    giou = []
    num = (atlbrs[:,0]).size
    x1 = atlbrs[:,0]
    y1 = atlbrs[:,1]
    x2 = atlbrs[:,2]
    y2 = atlbrs[:,3]

    xx1=btlbrs[:,0]
    yy1=btlbrs[:,1]
    xx2=btlbrs[:,2]
    yy2=btlbrs[:,3]

    area1 = (x2 -x1) * (y2 -y1)  #求取框的面积
    area2 = (xx2-xx1) * (yy2- yy1)
    for i in range (num):
        inter_max_x = np.minimum(x2[i], xx2[:])
        inter_max_y = np.minimum(y2[i], yy2[:])
        inter_min_x = np.maximum(x1[i], xx1[:])
        inter_min_y = np.maximum(y1[i], yy1[:])
        inter_w = np.maximum(0 ,inter_max_x-inter_min_x)
        inter_h = np.maximum(0 ,inter_max_y-inter_min_y)

        inter_areas = inter_w * inter_h

        out_max_x = np.maximum(x2[i], xx2[:])
        out_max_y = np.maximum(y2[i], yy2[:])
        out_min_x = np.minimum(x1[i], xx1[:])
        out_min_y = np.minimum(y1[i], yy1[:])
        out_w = np.maximum(0, out_max_x - out_min_x)
        out_h = np.maximum(0, out_max_y - out_min_y)

        outer_areas = out_w * out_h
        union = area1[i] + area2[:] - inter_areas
        ious = inter_areas / union
        gious = ious - (outer_areas - union)/outer_areas
        giou.append(gious)
    return giou

def iou_distance(atracks, btracks,type='iou'):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    if type=='iou':
        _ious = ious(atlbrs, btlbrs)
    elif type=='giou':
        _ious = giou(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

