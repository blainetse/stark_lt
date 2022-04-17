import numpy as np
import torch

import neuron.data as data
import neuron.data.transforms.pair_transforms.mmdet_transforms_multi as data_multi  # multi support_zs
import neuron.ops as ops
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.registry import DATASETS


__all__ = ['PairWrapper']


def _datasets(name):
    assert isinstance(name, str)
    if name == 'coco_train':
        return data.COCODetection(root_dir='/home/lc/Downloads/database/COCO-2017/train',subset='train')
    elif name == 'coco_val':
        return data.COCODetection(root_dir='/home/lc/Downloads/database/COCO-2017/val',subset='val')
    elif name == 'got10k_train':
        return data.GOT10k(root_dir='/home/lc/Downloads/database/GOT10K',subset='train')
    elif name == 'got10k_val':
        return data.GOT10k(root_dir='/home/lc/Downloads/database/GOT10K',subset='val')
    elif name == 'got10k_test':
        return data.GOT10k(root_dir='/home/lc/Downloads/database/GOT10K',subset='test')
    elif name == 'lasot_train':
        return data.LaSOT(root_dir='/home/lc/DataSSD/LaSOTBenchmark',subset='train')
    elif name == 'imagenet_vid':
        return data.ImageNetVID(subset=['train', 'val'])
    elif name == 'visdrone_vid':
       return  data.VisDroneVID(subset=['train', 'val'])
    elif name == 'fsod_train':    # added by lc
        roidb, ratio_list, ratio_index, cls_list, id_list = data.combined_roidb_for_training(
            root_dir='/home/lc/Downloads/database', subset='train', dataset_names=name, proposal_files=())
        info_list = np.concatenate([ratio_index[:, np.newaxis], cls_list[:, np.newaxis]], axis=1)
        info_list = np.concatenate([info_list, id_list[:, np.newaxis]], axis=1)
        dataset = data.RoiDataLoader(
            roidb,
            801,  # 801 model.classes, not used
            info_list,
            ratio_list,
            training=True)
        return dataset
    elif name == 'fsod_test':     # added by lc
        roidb, ratio_list, ratio_index, cls_list, id_list = data.combined_roidb_for_training(
            root_dir='/home/lc/Downloads/database', subset='test', dataset_names=name, proposal_files=())
        info_list = np.concatenate([ratio_index[:, np.newaxis], cls_list[:, np.newaxis]], axis=1)
        info_list = np.concatenate([info_list, id_list[:, np.newaxis]], axis=1)
        dataset = data.RoiDataLoader(
            roidb,
            201,  # 801 model.classes,  but not used
            info_list,
            ratio_list,
            training=False)
        return dataset
    elif name == 'tao_train':
        return data.Tao(annotation_path='/home/lc/Downloads/database/annotations-1.1/train.json', root_dir='/home/lc/Downloads/database/frames/train')
    else:
        raise KeyError('Unknown dataset:', name)

'''  single support_z  '''
def _transforms(name):
    # standard size: (1333, 800)
    if name == 'basic_train':
        return data.BasicPairTransforms(train=True)
    elif name == 'basic_test':
        return data.BasicPairTransforms(train=False)
    elif name == 'extra_partial':
        return data.ExtraPairTransforms(
        with_photometric=True,
        with_expand=False,
        with_crop=False)
    elif name == 'extra_full':
        return data.ExtraPairTransforms()
    else:
        raise KeyError('Unknown transform:', name)

''' multi support_zs '''
# def _transforms(name):
#     # standard size: (1333, 800)
#     if name == 'basic_train':
#         return data_multi.BasicPairTransforms_Multi(train=True)
#     elif name == 'basic_test':
#         return data_multi.BasicPairTransforms_Multi(train=False)
#     elif name == 'extra_partial':
#         return data_multi.ExtraPairTransforms_Multi(
#         with_photometric=True,
#         with_expand=False,
#         with_crop=False)
#     elif name == 'extra_full':
#         return data_multi.ExtraPairTransforms_Multi()
#     else:
#         raise KeyError('Unknown transform:', name)


@DATASETS.register_module
class PairWrapper(data.PairDataset):

    def __init__(self,
                 base_dataset='coco_train,got10k_train,lasot_train',
                 base_transforms='extra_partial',
                 sampling_prob=[0.4, 0.4, 0.2],
                 max_size=30000,
                 max_instances=8,
                 with_label=True,
                 **kwargs):
        # setup base dataset and indices (bounded by max_size)
        self.dataset_name = base_dataset  # for fsod
        self.base_dataset = self._setup_base_dataset(
            base_dataset, base_transforms, sampling_prob, max_size)
        self.indices = self._setup_indices(
            self.base_dataset, max_size)
        # member variables
        self.max_size = max_size
        self.max_instances = max_instances
        self.with_label = with_label
        self.flag = self.base_dataset.group_flags[self.indices]

    '''  single support_z  '''
    def __getitem__(self, index):
        if index == 0:
            self.indices = self._setup_indices(
                self.base_dataset, self.max_size)
        index = self.indices[index]
        item = self.base_dataset[index]  # x1,y1,x2,y2
        '''# self.base_dataset[index] : call the __getitem__ function of Image2Pair or Sequence2Pair or Oneshot2Pair'''
        # sanity check
        keys = [
            'img_z',
            'img_x',
            'gt_bboxes_z',
            'gt_bboxes_x',
            'img_meta_z',
            'img_meta_x']
        assert [k in item for k in keys]
        if not self.dataset_name=='fsod_train':
            assert len(item['gt_bboxes_z']) == len(item['gt_bboxes_x'])
        if len(item['gt_bboxes_z']) == 0 or \
            len(item['gt_bboxes_x']) == 0:
            return self._random_next()

        # if not self.dataset_name == 'fsod_train':
        #     # sample up to "max_instances" instances
        #     if self.max_instances > 0 and \
        #             len(item['gt_bboxes_z']) > self.max_instances:
        #         indices = np.random.choice(
        #             len(item['gt_bboxes_z']),
        #             self.max_instances,
        #             replace=False)
        #         item['gt_bboxes_z'] = item['gt_bboxes_z'][indices]
        #         item['gt_bboxes_x'] = item['gt_bboxes_x'][indices]
        #         # for class
        #         # item['cls'] = item['cls'][indices]  # for first stage : classification
        # else:  # for fsod_train
        #     if self.max_instances > 0 and \
        #             len(item['gt_bboxes_z']) > self.max_instances:
        #         indices = np.random.choice(
        #             len(item['gt_bboxes_z']),
        #             self.max_instances,
        #             replace=False)
        #         item['gt_bboxes_z'] = item['gt_bboxes_z'][indices]

        ''' joint training mode for coco '''
        # no max_instance for bboxes_x !!!  comment above !!!
        item['gt_z_cls'] = item['cls']
        if self.max_instances > 0 and \
                len(item['gt_bboxes_z']) > self.max_instances:
            indices = np.random.choice(
                len(item['gt_bboxes_z']),
                self.max_instances,
                replace=False)
            item['gt_bboxes_z'] = item['gt_bboxes_z'][indices]
            item['gt_z_cls'] = item['gt_z_cls'][indices]

        # construct DataContainer
        item = {
            'img_z': DC(item['img_z'], stack=True),
            'img_x': DC(item['img_x'], stack=True),
            'img_meta_z': DC(item['img_meta_z'], cpu_only=True),
            'img_meta_x': DC(item['img_meta_x'], cpu_only=True),
            'gt_bboxes_z': DC(item['gt_bboxes_z'].float()),
            'gt_bboxes_x': DC(item['gt_bboxes_x'].float()),
            # 'gt_z_cls': DC(torch.from_numpy(item['cls'])),   # for classifier(stage 1) and shot(not used)
            'gt_z_cls': DC(torch.from_numpy(item['gt_z_cls'])),   # joint training mode for coco
            'gt_x_cls': DC(torch.from_numpy(item['cls']))  # joint training mode for coco
        }  # for class

        # attach class labels if required to
        if self.with_label:
            _tmp = item['gt_bboxes_x'].data
            item['gt_labels'] = DC(_tmp.new_ones(len(_tmp)).long())

        return item

    ''' multi support_zs '''
    # def __getitem__(self, index):
    #     if index == 0:
    #         self.indices = self._setup_indices(
    #             self.base_dataset, self.max_size)
    #     index = self.indices[index]
    #     item = self.base_dataset[index]  # x1,y1,x2,y2
    #     '''# self.base_dataset[index] : call the __getitem__ function of Image2Pair or Sequence2Pair or Oneshot2Pair'''
    #     # sanity check
    #     keys = [
    #         'img_z',
    #         'img_x',
    #         'img_z_ne',
    #         'gt_bboxes_z',
    #         'gt_bboxes_x',
    #         'gt_bboxes_z_ne',
    #         'img_meta_z',
    #         'img_meta_x',
    #         'img_meta_z_ne']
    #     assert [k in item for k in keys]
    #     if not self.dataset_name == 'fsod_train':
    #         assert len(item['gt_bboxes_z']) == len(item['gt_bboxes_x'])
    #     if len(item['gt_bboxes_z']) == 0 or \
    #             len(item['gt_bboxes_x']) == 0 or \
    #             len(item['gt_bboxes_z_ne']) == 0:
    #         return self._random_next()
    #
    #     if not self.dataset_name == 'fsod_train':
    #         # sample up to "max_instances" instances
    #         if self.max_instances > 0 and \
    #                 len(item['gt_bboxes_z']) > self.max_instances:
    #             indices = np.random.choice(
    #                 len(item['gt_bboxes_z']),
    #                 self.max_instances,
    #                 replace=False)
    #             item['gt_bboxes_z'] = item['gt_bboxes_z'][indices]
    #             item['gt_bboxes_x'] = item['gt_bboxes_x'][indices]
    #             # for class
    #             # item['cls'] = item['cls'][indices]  # for first stage : classification
    #     else:  # for fsod_train
    #         if self.max_instances > 0 and \
    #                 len(item['gt_bboxes_z']) > self.max_instances:
    #             indices = np.random.choice(
    #                 len(item['gt_bboxes_z']),
    #                 self.max_instances,
    #                 replace=False)
    #             item['gt_bboxes_z'] = item['gt_bboxes_z'][indices]
    #
    #         if self.max_instances > 0 and \
    #                     len(item['gt_bboxes_z_ne']) > self.max_instances:
    #             indices = np.random.choice(
    #                 len(item['gt_bboxes_z_ne']),
    #                 self.max_instances,
    #                 replace=False)
    #             item['gt_bboxes_z_ne'] = item['gt_bboxes_z_ne'][indices]
    #
    #     # construct DataContainer
    #     item = {
    #         'img_z': DC(item['img_z'], stack=True),
    #         'img_x': DC(item['img_x'], stack=True),
    #         'img_z_ne': DC(item['img_z_ne'], stack=True),
    #         'img_meta_z': DC(item['img_meta_z'], cpu_only=True),
    #         'img_meta_x': DC(item['img_meta_x'], cpu_only=True),
    #         'img_meta_z_ne': DC(item['img_meta_z_ne'], cpu_only=True),
    #         'gt_bboxes_z': DC(item['gt_bboxes_z'].float()),
    #         'gt_bboxes_x': DC(item['gt_bboxes_x'].float()),
    #         'gt_bboxes_z_ne': DC(item['gt_bboxes_z_ne'].float()),
    #         # 'gt_z_cls': DC(torch.from_numpy(item['cls']))
    #     }  # for class
    #
    #     # attach class labels if required to
    #     if self.with_label:
    #         _tmp = item['gt_bboxes_x'].data
    #         item['gt_labels'] = DC(_tmp.new_ones(len(_tmp)).long())
    #
    #     return item

    def __len__(self):
        return len(self.indices)
    
    def _random_next(self):
        index = np.random.choice(len(self))
        return self.__getitem__(index)
    
    def _setup_indices(self, base_dataset, max_size):
        if max_size > 0 and len(base_dataset) > max_size:
            indices = np.random.choice(
                len(base_dataset), max_size, replace=False)
        else:
            indices = np.arange(len(base_dataset))
        return indices
    
    def _setup_base_dataset(self, base_dataset, base_transforms,
                            sampling_prob, max_size):
        names = base_dataset.split(',')
        datasets = []
        '''add fsod dataset'''
        for name in names:
            if 'coco' in name:
                # image-style dataset
                dataset = data.Image2Pair(
                    _datasets(name),
                    _transforms(base_transforms))
            elif 'fsod' in name:
                # image-style dataset
                dataset = data.Oneshot2Pair(
                    _datasets(name),
                    _transforms(base_transforms))
            else:
                # sequence-style dataset
                dataset = data.Seq2Pair(
                    _datasets(name),
                    _transforms(base_transforms))
            datasets.append(dataset)
        
        # concatenate datasets if necessary
        if len(datasets) == 1:
            return datasets[0]
        else:
            return data.RandomConcat(datasets, sampling_prob, max_size)
