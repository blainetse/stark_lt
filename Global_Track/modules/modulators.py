import torch.nn as nn
import torch
from mmdet.models.roi_extractors import SingleRoIExtractor, MultiRoIExtractor
from mmdet.core import bbox2roi
from mmcv.cnn import normal_init
from .cbam import CBAM
import torch.nn.functional as F


__all__ = ['RPN_Modulator', 'RCNN_Modulator','Cls_Modulator']


class RPN_Modulator(nn.Module):

    def __init__(self,
                 roi_out_size=7,
                 roi_sample_num=2,
                 channels=256,
                 strides=[4, 8, 16, 32],
                 featmap_num=5):
        super(RPN_Modulator, self).__init__()
        self.roi_extractor = SingleRoIExtractor(
            roi_layer={
                'type': 'RoIAlign',
                'out_size': roi_out_size,
                'sample_num': roi_sample_num},
            out_channels=channels,
            featmap_strides=strides)
        self.proj_modulator = nn.ModuleList([
            nn.Conv2d(channels, channels, roi_out_size, padding=0)
            for _ in range(featmap_num)])
        self.proj_out = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, padding=0)
            for _ in range(featmap_num)])
        # by lc
        self.channel_weight = CBAM(gate_channels=256, no_spatial=False)
        # self.channel_weight = nn.ModuleList([
        #     CBAM(gate_channels=256, no_spatial=False)
        #     for _ in range(featmap_num)])

        # end

    
    def forward(self, feats_z, feats_x, gt_bboxes_z):
        return self.inference(
            feats_x,
            modulator=self.learn(feats_z, gt_bboxes_z))
    
    def inference(self, feats_x, modulator):
        # print(len(feats_x))
        n_imgs = len(feats_x[0])
        for i in range(n_imgs):
            n_instances = len(modulator[i])
            for j in range(n_instances):
                query = modulator[i][j:j + 1]
                gallary = [f[i:i + 1] for f in feats_x]
                out_ij = [self.proj_modulator[k](query) * gallary[k]
                          for k in range(len(gallary))]
                # out_ij = [self.xcorr_depthwise(gallary[k],self.proj_modulator[k](query))  # 感觉效果是一样的,1*1element 不就是depthwise没有1*1卷积,后面projout就齐了
                #           for k in range(len(gallary))]    # by lc
                out_ij = [p(o) for p, o in zip(self.proj_out, out_ij)]
                # out_ij = [p(o) for p, o in zip(self.channel_weight, out_ij)] # by lc
                # out_ij = [self.channel_weight(o)  for o in out_ij]  # by lc
                yield out_ij, i, j

    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        import torch.nn.functional as F
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def learn(self, feats_z, gt_bboxes_z):
        rois = bbox2roi(gt_bboxes_z)
        # print(self.roi_extractor.num_inputs)
        # # print(len(rois[0]))  # 5 是因为第一个是图片的序号，多张图输入时
        # print(len(feats_z))
        # print(feats_z[0].shape)
        # print(feats_z[1].shape)
        # print(feats_z[2].shape)
        # print(feats_z[3].shape)
        # print(feats_z[4].shape)
        # print('###############')

        bbox_feats = self.roi_extractor(
            feats_z[:self.roi_extractor.num_inputs], rois)
        bbox_feats = self.channel_weight(bbox_feats)  # 唯一的改动
        # print(bbox_feats[0].shape)
        # print('*************')
        modulator = [bbox_feats[rois[:, 0] == j]
                     for j in range(len(gt_bboxes_z))]
        return modulator

    '''add reid loss'''
    # def metric_net(self,feats_z, feats_x, gt_bboxes_z, gt_bboxes_x):


    def init_weights(self):
        for m in self.proj_modulator:
            normal_init(m, std=0.01)
        for m in self.proj_out:
            normal_init(m, std=0.01)


class RCNN_Modulator(nn.Module):

    def __init__(self, channels=256):
        super(RCNN_Modulator, self).__init__()
        self.proj_z = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_x = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, z, x):
        return self.inference(x, self.learn(z))

    def inference(self, x, modulator):
        # assume one image and one instance only
        assert len(modulator) == 1
        return self.proj_out(self.proj_x(x) * modulator)
    
    def learn(self, z):
        # assume one image and one instance only
        assert len(z) == 1
        return self.proj_z(z)
    
    def init_weights(self):
        normal_init(self.proj_z, std=0.01)
        normal_init(self.proj_x, std=0.01)
        normal_init(self.proj_out, std=0.01)



'''encode class information to be an attention map'''
'''pseudo code:
        input:  embeddings, shape of (batchsize,256,1,1)
                feats_x, 5-element list, (batchsize,256,h,w)
        output: attention_map, shape of (batchsize,1,h,w)
                out, shape of (batchsize,256,h,w)
                
'''
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class Cls_Modulator(nn.Module):
    def __init__(self,
                 roi_out_size=7,
                 roi_sample_num=2,
                 channels=256,
                 strides=[4, 8, 16, 32],
                 featmap_num=5):
        super(Cls_Modulator, self).__init__()
        # self.roi_extractor = SingleRoIExtractor(
        #     roi_layer={
        #         'type': 'RoIAlign',
        #         'out_size': roi_out_size,
        #         'sample_num': roi_sample_num},
        #     out_channels=channels,
        #     featmap_strides=strides,
        #     finest_scale=28)     # for atss , p3-p6   56
        self.roi_extractor = MultiRoIExtractor(
            roi_layer={
                'type': 'RoIAlign',
                'out_size': roi_out_size,
                'sample_num': roi_sample_num},
            out_channels=channels,
            featmap_strides=strides,
            finest_scale=28)
        '''attention map'''
        self.pointwise_conv = nn.ModuleList([
            nn.Conv2d(channels, 1, 1, padding=0)
            for _ in range(featmap_num)])
        # self.pointwise_conv = nn.ModuleList([
        #     nn.Conv2d(channels, 1, 3, padding=1)
        #     for _ in range(featmap_num)])
        self.embeddings = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, padding=0),
        ])
        # self.embeddings = nn.ModuleList([
        #     nn.Conv2d(channels, channels, 3, padding=0),
        #     nn.Conv2d(channels, channels, 3, padding=0), # 3,3
        # ])
        self.proj_em = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, padding=0)
            for _ in range(featmap_num)])
        ''' instance specific modulation'''
        self.proj_modulator = nn.ModuleList([
            nn.Conv2d(channels, channels, roi_out_size, padding=0)  # nn.Conv2d(channels, channels, roi_out_size-2, padding=0)
            for _ in range(featmap_num)])
        self.proj_out = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, padding=0)
            for _ in range(featmap_num)])
        # by lc attention
        self.channel_weight = CBAM(gate_channels=256, no_spatial=False)

        # self.proj_out_ = nn.ModuleList([
        #     nn.Conv2d(channels, 1, 1, padding=0)
        #     for _ in range(featmap_num)])
        self.proj_out_ = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels, 1, 1, padding=0),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
        )  for _ in range(featmap_num)])
        self.proj_3 = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1)
            for _ in range(featmap_num)])
        # self.proj_modulator_ = nn.ModuleList([
        #     # nn.Conv2d(channels, channels, roi_out_size, padding=0)
        #     nn.Conv2d(channels, channels, roi_out_size-2, padding=0)
        #     for _ in range(featmap_num)])
        # self.channel_weight_fusion = CBAM(gate_channels=512, no_spatial=True)

        # self.mlp = nn.ModuleList([nn.Sequential(
        #     Flatten(),
        #     nn.Linear(channels, channels // 16),
        #     nn.ReLU(),
        #     nn.Linear(channels // 16, channels)
        # )  for _ in range(featmap_num)])
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channels, channels // 16),
            nn.ReLU(),
            nn.Linear(channels // 16, channels)
        )
        self.proj_modulator_spatial = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, padding=0)
            for _ in range(featmap_num)])
        # self.proj_modulator_cls = nn.ModuleList([
        #     nn.Conv2d(channels, channels, 1, padding=0)
        #     for _ in range(featmap_num)])
        self.proj_modulator_cls = nn.ModuleList([
            nn.Conv2d(channels, channels, roi_out_size, padding=0)
            for _ in range(featmap_num)])
        self.proj_out_cls = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, padding=0)
            for _ in range(featmap_num)])
        self.proj_embed = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=0),  # 5*5
            nn.Conv2d(channels, channels, 3, padding=0)  # 3*3
            ])
        self.proj_spatial = nn.ModuleList([
            nn.Conv2d(channels, channels, roi_out_size-2, padding=0)  # (3,3)
            for _ in range(featmap_num)])
        self.proj_spatial_out = nn.ModuleList([
            nn.Conv2d(channels, 1, 3, padding=1)  # (3,3)
            for _ in range(featmap_num)])

    # def forward(self,IDnet, feats_z, feats_x, gt_bboxes_z,gt_z_cls, track_flag, joint_flag=False,switch_up=True):
    def forward(self,IDnet, feats_z, feats_x, gt_bboxes_z, track_flag, joint_flag=False,switch_up=True):
    # def forward(self, feats_z, feats_x, gt_bboxes_z, track_flag, joint_flag=False,switch_up=True):
        # 首先要将gt_bboxes_z裁成每张图，每一个物体，也就是i,j，产生一个对应的embeddings
        # embeddings和feats_x进一步调用xcorr_depthwise生成attention map
        # attention map和feats_x再进一步生成out_ij,i,j
        # self.IDnet = IDnet  # IDNet is added after  2020.10.7
        self.gt_bboxes_z = gt_bboxes_z
        ''' for instance speicific modlator  for sot tracking '''
        modulator = self.learn_rpn_modulator(feats_z, gt_bboxes_z)

        # return self.inference(
        #   feats_x,
        #    query_feats=self.learn(feats_z, gt_bboxes_z), modulator=modulator, gt_z_cls=gt_z_cls, track_flag=track_flag, joint_flag=joint_flag,switch_up=switch_up)

        return self.inference(
            feats_x,
            query_feats=self.learn(feats_z, gt_bboxes_z), modulator=modulator, track_flag=track_flag,
            joint_flag=joint_flag, switch_up=switch_up)
        # return self.inference(
        #     feats_x,
        #     query_feats=self.learn(feats_z, gt_bboxes_z), modulator=None, track_flag=track_flag)


    # def inference(self, feats_x, query_feats, modulator,gt_z_cls, track_flag,joint_flag,switch_up ):
    def inference(self, feats_x, query_feats, modulator, track_flag,joint_flag,switch_up ):
        n_imgs = len(feats_x[0])
        for i in range(n_imgs):
            n_instances = len(query_feats[i])
            for j in range(n_instances):
                query = query_feats[0][i][j:j + 1]  # query应该是1,256,7,7  -> 4,256,7,7
                query_sin = query_feats[1][i][j:j + 1]  # query_sin应该是1,256,7,7
                gallary = [f[i:i + 1] for f in feats_x]
                embedings = self.generate_embeddings(query_sin)  # 1,256,1,1 所有层是共用的
                embedings_ = nn.functional.adaptive_avg_pool2d(embedings, (1, 1))

                #[-----------------------------------------------]
                # if gt_z_cls[i][j:j + 1].item()>=0.0 and gt_z_cls[i][j:j + 1].item()<=79.0:
                #    switch_up = True
                # else:
                #     switch_up = False
                #___________________________________________________

                # [-------------------------cls--------------------------]
                # id_pred = self.IDnet[0](embedings_.view(1, -1))
                # id_pred = self.IDnet[1](id_pred)
                # id_pred_ = nn.functional.softmax(id_pred)
                # # # if torch.argmax(id_pred_[0, :]).item() <= 79 and torch.argmax(id_pred_[0, :]).item() >= 0:
                # # #     switch_up = True
                # # # else:
                # # #     switch_up = False
                # #
                # if max(id_pred_[0,:])>0.7:
                #     switch_up = True
                # else:
                #     switch_up = False
                # #
                # print(switch_up)
                # switch_up = False
                # print(max(id_pred_[0,:]))
                # print(torch.argmax(id_pred_[0, :]))
                # [------------------------------------------------------]

                # id_pred = self.IDnet[0](embedings_.view(1, -1))
                # id_pred_ = nn.functional.softmax(id_pred)
                # if max(id_pred_[0, :])>0.5:
                #     switch_up = True
                # else:
                #     switch_up = False
                # print(max(id_pred_[0, :]))
                # print(torch.argmax(id_pred_[0, :]))
                # embedings_ = embedings
                # print(embedings.shape)
                ''' all the operations is below '''

                ''' for test '''
                # print(track_flag)
                if joint_flag:
                    out_ij_shot = gallary
                ''' instance specific modulation for sot tracking '''
                if track_flag:
                    # pass
                    #########################################################################################################################
                    ''' version spatial weight '''
                    attention_list = [self.xcorr_depthwise(gallary[k], nn.functional.adaptive_avg_pool2d(query[0,k:k+1,...], (5, 5)))
                                      # gallary的每个元素都是其中一个特征层，feats_x本来就是5元素的list，每个元素是batch大小，batch为图像个数，i作指示
                                      for k in range(len(gallary))]
                    # attention_list = [p(o) for p, o in zip(self.proj_out_, attention_list)]
                    attention_list = [self.proj_out_[k](attention_list[k])  for k in range(len(attention_list))]
                    m = nn.Sigmoid()
                    attention_map_list = [m(attention_list[k]) for k in range(len(attention_list))]

                    # corr_emd = nn.functional.adaptive_avg_pool2d(query, (3, 3))  # (3, 3)
                    # # corr_emd = nn.functional.adaptive_max_pool2d(query, (5, 5))  # (3, 3)
                    # # corr_emd = query
                    # # attention_list = [self.xcorr_depthwise(gallary[k], self.proj_modulator_spatial[k](corr_emd))
                    # #                   # gallary的每个元素都是其中一个特征层，feats_x本来就是5元素的list，每个元素是batch大小，batch为图像个数，i作指示
                    # #                   for k in range(len(gallary))]
                    # attention_list = [self.xcorr_depthwise(gallary[k], corr_emd)
                    #                   # gallary的每个元素都是其中一个特征层，feats_x本来就是5元素的list，每个元素是batch大小，batch为图像个数，i作指示
                    #                   for k in range(len(gallary))]
                    # attention_list = [p(o) for p, o in zip(self.proj_out_, attention_list)]
                    # m = nn.Sigmoid()
                    # attention_map_list = [m(attention_list[k]) for k in range(len(attention_list))]
                    # import numpy as np
                    # import cv2
                    # heatmap = attention_map_list[0][0, ...].cpu().permute(1, 2, 0).numpy() * 255
                    # size = heatmap.shape[0:2]
                    # from skimage import transform
                    # heatmap = transform.resize(heatmap, (heatmap.shape[0]*2, heatmap.shape[1]*2))
                    # heatmap0 = heatmap.astype(np.uint8)
                    # heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET) # HOT
                    # cv2.imshow('heatmap', heatmap)
                    # cv2.waitKey(1)
                    #
                    # heatmap = attention_map_list[1][0, ...].cpu().permute(1, 2, 0).numpy() * 255
                    # size = heatmap.shape[0:2]
                    # from skimage import transform
                    # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 4, heatmap.shape[1] * 4))
                    # heatmap0 = heatmap.astype(np.uint8)
                    # heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # HOT
                    # cv2.imshow('heatmap1', heatmap)
                    # cv2.waitKey(1)
                    #
                    # heatmap = attention_map_list[2][0, ...].cpu().permute(1, 2, 0).numpy() * 255
                    # size = heatmap.shape[0:2]
                    # from skimage import transform
                    # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 8, heatmap.shape[1] * 8))
                    # heatmap0 = heatmap.astype(np.uint8)
                    # heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # HOT
                    # cv2.imshow('heatmap2', heatmap)
                    # cv2.waitKey(1)
                    #
                    # heatmap = attention_map_list[3][0, ...].cpu().permute(1, 2, 0).numpy() * 255
                    # size = heatmap.shape[0:2]
                    # from skimage import transform
                    # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 16, heatmap.shape[1] * 16))
                    # heatmap0 = heatmap.astype(np.uint8)
                    # heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # HOT
                    # cv2.imshow('heatmap3', heatmap)
                    # cv2.waitKey(1)
                    # # #
                    # # # # def gaussian2D(shape, sigma=1):
                    # # # #     m, n = [(ss - 1.) / 2. for ss in shape]
                    # # # #     y, x = np.ogrid[-m:m + 1, -n:n + 1]
                    # # # #
                    # # # #     h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
                    # # # #     h[h < np.finfo(h.dtype).eps * h.max()] = 0
                    # # # #     return h
                    # # # # def draw_gaussian(heatmap, center, sigma):
                    # # # #     tmp_size = sigma * 3
                    # # # #     mu_x = int(center[0] + 0.5)
                    # # # #     mu_y = int(center[1] + 0.5)
                    # # # #     w, h = heatmap.shape[0], heatmap.shape[1]
                    # # # #     ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    # # # #     br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    # # # #     if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
                    # # # #         return heatmap
                    # # # #     size = 2 * tmp_size + 1
                    # # # #     x = np.arange(0, size, 1, np.float32)
                    # # # #     y = x[:, np.newaxis]
                    # # # #     x0 = y0 = size // 2
                    # # # #     g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
                    # # # #     g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
                    # # # #     g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
                    # # # #     img_x = max(0, ul[0]), min(br[0], h)
                    # # # #     img_y = max(0, ul[1]), min(br[1], w)
                    # # # #     heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                    # # # #         heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
                    # # # #         g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                    # # # #     return heatmap
                    # # # #
                    # # # # heatmap_ = gaussian2D(heatmap.shape[0:2])
                    # # # # heatmap = draw_gaussian(heatmap_,self.gt_bboxes_z[0][0][0:2]/4,10)*255
                    # # # # heatmap = (heatmap).astype(np.uint8)
                    # # # # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # HOT
                    # # # # cv2.imshow('heatmap-gaussian', heatmap)
                    # # # # cv2.waitKey(1)
                    # # #
                    # # # heatmap = attention_map_list[1][0, ...].cpu().permute(1, 2, 0).numpy() * 255
                    # # # from skimage import transform
                    # # # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 4, heatmap.shape[1] * 4))
                    # # # heatmap1 = heatmap.astype(np.uint8)
                    # # # heatmap = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)  # HOT
                    # # # cv2.imshow('heatmap-1', heatmap)
                    # # # cv2.waitKey(1)
                    # # #
                    # # # heatmap = attention_map_list[2][0, ...].cpu().permute(1, 2, 0).numpy() * 255 # *0.2
                    # # # from skimage import transform
                    # # # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 8, heatmap.shape[1] * 8))
                    # # # heatmap2 = heatmap.astype(np.uint8)
                    # # # heatmap = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)  # HOT
                    # # # cv2.imshow('heatmap-2', heatmap)
                    # # # cv2.waitKey(1)
                    # # #
                    # # # heatmap = attention_map_list[3][0, ...].cpu().permute(1, 2, 0).numpy() * 255
                    # # # from skimage import transform
                    # # # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 16, heatmap.shape[1] * 16))
                    # # # heatmap3 = heatmap.astype(np.uint8)
                    # # # heatmap = cv2.applyColorMap(heatmap3, cv2.COLORMAP_JET)  # HOT
                    # # # cv2.imshow('heatmap-3', heatmap)
                    # # # cv2.waitKey(1)
                    # # #
                    # # # hanning1 = np.hanning(size[0]*2)
                    # # # hanning2 = np.hanning(size[1]*2)
                    # # # heatmap_hn = np.outer(hanning1, hanning2)*255
                    # # # heatmap_hn = heatmap_hn.astype(np.uint8)
                    # # # heatmap_hn = cv2.applyColorMap(heatmap_hn, cv2.COLORMAP_JET)  # HOT
                    # # # cv2.imshow('heatmap-hanning', heatmap_hn)
                    # # # cv2.waitKey(1)
                    # # # #
                    # # # # heatmap = attention_map_list[4][0, ...].cpu().permute(1, 2, 0).numpy() * 255
                    # # # # from skimage import transform
                    # # # # heatmap = transform.resize(heatmap, (heatmap.shape[0] * 8, heatmap.shape[1] * 8))
                    # # # # heatmap4 = heatmap.astype(np.uint8)
                    # # # # heatmap = cv2.applyColorMap(heatmap4, cv2.COLORMAP_JET)  # HOT
                    # # # # cv2.imshow('heatmap-4', heatmap)
                    # # # # cv2.waitKey(1)
                    # # #
                    # # # # heatmap = np.stack((heatmap0,heatmap1,heatmap2,heatmap3),axis=2)  # h,w,4,1
                    # # # # heatmap = np.max(heatmap[...,0],axis=-1)
                    # # # # heatmap = cv2.applyColorMap(((heatmap/np.max(heatmap))*255).astype(np.uint8), cv2.COLORMAP_JET)  # HOT
                    # # # # cv2.imshow('heatmap-sum', heatmap)
                    # # # # cv2.waitKey(1)
                    out_ij = [p * o for p, o in zip(gallary, attention_map_list)]
                    #########################################################################################################################
                    ''' version spatial weight 10.29'''
                    # corr_emd = query
                    # attention_list = [self.xcorr_depthwise(gallary[k],  self.proj_spatial[k](corr_emd))
                    #                   # gallary的每个元素都是其中一个特征层，feats_x本来就是5元素的list，每个元素是batch大小，batch为图像个数，i作指示
                    #                   for k in range(len(gallary))]
                    # attention_list = [p(o) for p, o in zip(self.proj_spatial_out, attention_list)]
                    # m = nn.Sigmoid()
                    # attention_map_list = [m(attention_list[k]) for k in range(len(attention_list))]
                    # ############# [ concat attentions ] ############
                    att0 = attention_map_list[0]
                    att1 = nn.functional.interpolate(attention_map_list[1], size=att0.shape[2:])
                    att2 = nn.functional.interpolate(attention_map_list[2], size=att0.shape[2:])
                    att3 = nn.functional.interpolate(attention_map_list[3], size=att0.shape[2:])
                    att = torch.cat(([att0,att1,att2,att3]),dim=1)  # (1,4,h,w)
                    att = torch.max(att,dim=1)[0].unsqueeze(dim=1)  # (1,1,h,w)
                    # gt_mask = 1,1,h,w  # 0.9/0.8/0.7
                    # out_ij = [p * o for p, o in zip(gallary, attention_map_list)]
                    #############[ original modulate ]############
                    # out_ij = gallary
                    query_modulator = modulator[i][j:j + 1]
                    gallary_cls_ = [f[i:i + 1] for f in out_ij]
                    out_ij = [self.proj_modulator[k](query_modulator) * gallary_cls_[k]
                              for k in range(len(gallary_cls_))]
                    '''for conv3*3(roi_size-2) version'''
                    # out_ij = [self.xcorr_depthwise(gallary_cls_[k],self.proj_modulator[k](query_modulator))
                    #           for k in range(len(gallary))]    # by lc
                    '''for conv5*5(roi_size-4) version'''
                    # out_ij = [self.xcorr_depthwise(gallary_cls_[k], self.proj_modulator[k](query_modulator))
                    #           for k in range(len(gallary))]  # by lc

                    out_ij = [p(o) for p, o in zip(self.proj_out, out_ij)]
                    # #
                #
                # yield embedings_, out_ij, i, j
                if joint_flag:
                    out_ij_track = out_ij
                    # import matplotlib
                    # # matplotlib.use('TkAgg')
                    # import matplotlib.pyplot
                    # import matplotlib.pyplot as plt
                    # plt.imshow(attention_map_list[0][0,...].cpu().permute(1,2,0))
                    # plt.savefig('last.jpg')
                    yield embedings_,att, out_ij_track, out_ij_shot, i, j

                # yield embedings_, out_ij, i, j   # for special reason, comment
                # yield None, out_ij, i, j   # for special reason, comment


    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
            # out is shape of (batchsize,256,h,w) list of 5
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        import torch.nn.functional as F
        # out = F.conv2d(x, kernel, groups=batch*channel,padding=1)  # for 3*3
        out = F.conv2d(x, kernel, groups=batch*channel,padding=2)  # for 5*5
        # out = F.conv2d(x, kernel, groups=batch*channel,padding=3)  # for 7*7
        # out = F.conv2d(x, kernel, groups=batch*channel)   # for 1*1
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def xcorr_depthwise_1(self, x, kernel):
        """depthwise cross correlation
            # out is shape of (batchsize,256,h,w) list of 5
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        import torch.nn.functional as F
        # out = F.conv2d(x, kernel, groups=batch*channel,padding=1)  # for 3*3
        # out = F.conv2d(x, kernel, groups=batch*channel,padding=2)  # for 5*5
        # out = F.conv2d(x, kernel, groups=batch*channel,padding=3)  # for 7*7
        out = F.conv2d(x, kernel, groups=batch*channel)   # for 1*1
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out


    def xcorr_fast(self, x, kernel):
        """group conv2d to calculate cross correlation, fast version
        """
        batch = kernel.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(1, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po

    def learn(self, feats_z, gt_bboxes_z):
        rois = bbox2roi(gt_bboxes_z)
        # bbox_feats = self.roi_extractor(
        #     feats_z[:self.roi_extractor.num_inputs],rois)
        # query_feats = [bbox_feats[rois[:, 0] == j]
        #                for j in range(len(gt_bboxes_z))]
        # return query_feats
        bbox_feats_multi, bbox_feats_single = self.roi_extractor(
            feats_z[:self.roi_extractor.num_inputs], rois)
        query_feats_multi = [bbox_feats_multi[:,rois[:, 0] == j,...].permute(1,0,2,3,4)
                       for j in range(len(gt_bboxes_z))]
        query_feats_single = [bbox_feats_single[rois[:, 0] == j]
                              for j in range(len(gt_bboxes_z))]
        return query_feats_multi, query_feats_single  # 4,n,256,7,7

    ''' query: (1,256,7,7)
        output: (1,256,1,1)'''
    def generate_embeddings(self,query):
        embeddings = query
        for p in self.embeddings:
            embeddings = p(embeddings)
        return embeddings

    def learn_rpn_modulator(self, feats_z, gt_bboxes_z):
        rois = bbox2roi(gt_bboxes_z)
        _,bbox_feats = self.roi_extractor(
            feats_z[:self.roi_extractor.num_inputs], rois)
        # bbox_feats = self.roi_extractor(
        #     feats_z[:self.roi_extractor.num_inputs], rois)
        bbox_feats = self.channel_weight(bbox_feats)  # 唯一的改动
        # print(bbox_feats[0].shape)
        # print('*************')
        modulator = [bbox_feats[rois[:, 0] == j]
                     for j in range(len(gt_bboxes_z))]
        return modulator

    def init_weights(self):
        for m in self.pointwise_conv:
            normal_init(m, std=0.01)
        for m in self.embeddings:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.proj_em:
            normal_init(m, std=0.01)
        ''' instance specific modulation '''
        for m in self.proj_modulator:
            normal_init(m, std=0.01)
        for m in self.proj_out:
            normal_init(m, std=0.01)
        for m in self.proj_modulator_spatial:
            normal_init(m, std=0.01)
        for m in self.proj_modulator_cls:
            normal_init(m, std=0.01)
        for m in self.proj_embed:
            normal_init(m, std=0.01)

        # for m in self.con3_3:
        #     normal_init(m, std=0.01)
        # for m in self.conv1_1:
        #     normal_init(m, std=0.01)
        # for m in self.fusion:
        #     normal_init(m, std=0.01)
        #
        # for m in self.proj_modulator_:
        #     normal_init(m, std=0.01)
        for m in self.proj_out_:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.proj_3:
            normal_init(m, std=0.01)
        for m in self.proj_out_cls:
            normal_init(m, std=0.01)

    def freeze(self):
        for para in self.pointwise_conv.parameters():
            para.requires_grad = False
        for para in self.embeddings.parameters():
            para.requires_grad = False
        for para in self.proj_em.parameters():
            para.requires_grad = False
        for para in self.mlp.parameters():
            para.requires_grad = False
        for para in self.proj_embed.parameters():
            para.requires_grad = False
        for para in self.proj_3.parameters():
            para.requires_grad = False
        for para in self.proj_out_cls.parameters():
            para.requires_grad = False

