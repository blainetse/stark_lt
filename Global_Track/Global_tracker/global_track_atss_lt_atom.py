# import torch
# import numpy as np
# import torch.nn.functional as F
# import torch.nn
# import importlib
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
# ''' pytracking '''
# import math
# import time
# from .pytracking.libs import dcf, fourier, TensorList, operation
# from .pytracking.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2
# from .pytracking.optim import ConvProblem, FactorizedConvProblem
# from .pytracking.features import augmentation
# from .pytracking.parameter.atom.default_vot import parameters
#
# __all__ = ['GlobalATSSTrackVOT_ATOM']
#
#
# class GlobalATSSTrackVOT_ATOM(TrackerVOT):
#
#     # def __init__(self, cfg_file, ckp_file, ckp_file_det, transforms, name_suffix=''):
#     def __init__(self, cfg_file, ckp_file, transforms, name_suffix=''):
#         name = 'GlobalATSSTrackVOT_ATOM'
#         if name_suffix:
#             name += '_' + name_suffix
#         super(GlobalATSSTrackVOT_ATOM, self).__init__(
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
#         ''' atom '''
#         self.params = self.get_parameters()
#         # Get feature specific params
#         self.fparams = self.params.features.get_fparams('feature_params')
#
#         state = np.array([bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]])
#         # Initialize features
#         # self.initialize_features()
#
#         # Get position and size
#         self.pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
#         self.target_sz = torch.Tensor([state[3], state[2]])
#
#         # Set search area
#         self.target_scale = 1.0
#         search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
#         if search_area > self.params.max_image_sample_size:
#             self.target_scale = math.sqrt(search_area / self.params.max_image_sample_size)
#         elif search_area < self.params.min_image_sample_size:
#             self.target_scale = math.sqrt(search_area / self.params.min_image_sample_size)
#
#         # Target size in base scale
#         self.base_target_sz = self.target_sz / self.target_scale
#
#         # Use odd square search area and set sizes
#         feat_max_stride = max(self.params.features.stride())
#         if self.params.get('search_area_shape', 'square') == 'square':
#             self.img_sample_sz = torch.round(
#                 torch.sqrt(torch.prod(self.base_target_sz * self.params.search_area_scale))) * torch.ones(2)
#         elif self.params.search_area_shape == 'initrect':
#             self.img_sample_sz = torch.round(self.base_target_sz * self.params.search_area_scale)
#         else:
#             raise ValueError('Unknown search area shape')
#         if self.params.feature_size_odd:
#             self.img_sample_sz += feat_max_stride - self.img_sample_sz % (2 * feat_max_stride)
#         else:
#             self.img_sample_sz += feat_max_stride - (self.img_sample_sz + feat_max_stride) % (2 * feat_max_stride)
#
#         # Set sizes
#         self.img_support_sz = self.img_sample_sz
#         self.feature_sz = self.params.features.size(self.img_sample_sz)
#         self.output_sz = self.params.score_upsample_factor * self.img_support_sz  # Interpolated size of the output
#         self.kernel_size = self.fparams.attribute('kernel_size')
#
#         # Optimization options
#         self.params.precond_learning_rate = self.fparams.attribute('learning_rate')
#         if self.params.CG_forgetting_rate is None or max(self.params.precond_learning_rate) >= 1:
#             self.params.direction_forget_factor = 0
#         else:
#             self.params.direction_forget_factor = (1 - max(
#                 self.params.precond_learning_rate)) ** self.params.CG_forgetting_rate
#
#         self.output_window = None
#         if self.params.get('window_output', False):
#             if self.params.get('use_clipped_window', False):
#                 self.output_window = dcf.hann2d_clipped(self.output_sz.long(),
#                                                         self.output_sz.long() * self.params.effective_search_area / self.params.search_area_scale,
#                                                         centered=False).to(self.params.device)
#             else:
#                 self.output_window = dcf.hann2d(self.output_sz.long(), centered=False).to(self.params.device)
#
#
#         # Initialize some learning things
#         self.init_learning()
#
#         # Extract and transform sample
#         x = self.generate_init_samples(img_init)
#
#         # Transform to get the training sample
#         train_x = self.preprocess_sample(x)
#
#         # Generate label function
#         init_y = self.init_label_function(train_x)
#
#         # Init memory
#         self.init_memory(train_x)
#
#         # Init optimizer and do initial optimization
#         self.init_optimization(train_x, init_y)
#
#
#         return times
#
#
#     def update(self, img, gt, **kwargs):
#         with torch.no_grad():
#             self.model.eval()
#
#             # prepare gallary data
#             img_meta = {'ori_shape': img.shape}
#             img_original = img
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
#         ''' pytracking '''
#         # Get sample
#         sample_pos = self.pos.round()
#         sample_scales = self.target_scale * self.params.scale_factors
#         test_x = self.extract_processed_sample(img_original, self.pos, sample_scales, self.img_sample_sz)
#
#         # Compute scores
#         scores_raw = self.apply_filter(test_x)
#         translation_vec, scale_ind, s, flag = self.localize_target(scores_raw)
#         # Update position and scale
#         if flag != 'not_found':
#             if self.use_iou_net:
#                 update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
#                 if self.params.get('use_classifier', True):
#                     self.update_state(sample_pos + translation_vec)
#                 self.refine_target_box(sample_pos, sample_scales[scale_ind], scale_ind, update_scale_flag)
#             elif self.params.get('use_classifier', True):
#                 self.update_state(sample_pos + translation_vec, sample_scales[scale_ind])
#
#         score_map = s[scale_ind, ...]
#         max_score = torch.max(score_map).item()
#
#         # ------- UPDATE ------- #
#
#         # Check flags and set learning rate if hard negative
#         update_flag = flag not in ['not_found', 'uncertain']
#         hard_negative = (flag == 'hard_negative')
#         learning_rate = self.params.hard_negative_learning_rate if hard_negative else None
#
#         if update_flag:
#             # Get train sample
#             train_x = TensorList([x[scale_ind:scale_ind + 1, ...] for x in test_x])
#
#             # Create label for sample
#             train_y = self.get_label_function(sample_pos, sample_scales[scale_ind])
#
#             # Update memory
#             self.update_memory(train_x, train_y, learning_rate)
#
#         # Train filter
#         if hard_negative:
#             self.filter_optimizer.run(self.params.hard_negative_CG_iter)
#         elif (self.frame_num - 1) % self.params.train_skipping == 0:
#             self.filter_optimizer.run(self.params.CG_iter)
#
#
#         # if not kwargs.get('return_all', False):
#         #     # return the top-1 detection
#         #     max_ind = results[:, -1].argmax()
#         #     # return results[max_ind, :4]    # 不是返回所有的，就是返回最大值
#         #     return results[max_ind, :4], results[max_ind, 4]    # 如果是vot 18lt
#         # else:
#         #     # return all detections
#         #     return results
#
#     def init_optimization(self, train_x, init_y):
#         # Initialize filter
#         filter_init_method = self.params.get('filter_init_method', 'zeros')
#         self.filter = TensorList(
#             [x.new_zeros(1, cdim, sz[0], sz[1]) for x, cdim, sz in zip(train_x, self.compressed_dim, self.kernel_size)])
#         if filter_init_method == 'zeros':
#             pass
#         elif filter_init_method == 'randn':
#             for f in self.filter:
#                 f.normal_(0, 1/f.numel())
#         else:
#             raise ValueError('Unknown "filter_init_method"')
#
#         # Get parameters
#         self.params.update_projection_matrix = self.params.get('update_projection_matrix', True) and self.params.use_projection_matrix
#         optimizer = self.params.get('optimizer', 'GaussNewtonCG')
#
#         # Setup factorized joint optimization
#         if self.params.update_projection_matrix:
#             self.joint_problem = FactorizedConvProblem(self.init_training_samples, init_y, self.filter_reg,
#                                                        self.fparams.attribute('projection_reg'), self.params, self.init_sample_weights,
#                                                        self.projection_activation, self.response_activation)
#
#             # Variable containing both filter and projection matrix
#             joint_var = self.filter.concat(self.projection_matrix)
#
#             # Initialize optimizer
#             analyze_convergence = self.params.get('analyze_convergence', False)
#             if optimizer == 'GaussNewtonCG':
#                 self.joint_optimizer = GaussNewtonCG(self.joint_problem, joint_var, debug=(self.params.debug >= 1),
#                                                      plotting=(self.params.debug >= 3), analyze=analyze_convergence,
#                                                      visdom=self.visdom)
#             elif optimizer == 'GradientDescentL2':
#                 self.joint_optimizer = GradientDescentL2(self.joint_problem, joint_var, self.params.optimizer_step_length, self.params.optimizer_momentum, plotting=(self.params.debug >= 3), debug=(self.params.debug >= 1),
#                                                          visdom=self.visdom)
#
#             # Do joint optimization
#             if isinstance(self.params.init_CG_iter, (list, tuple)):
#                 self.joint_optimizer.run(self.params.init_CG_iter)
#             else:
#                 self.joint_optimizer.run(self.params.init_CG_iter // self.params.init_GN_iter, self.params.init_GN_iter)
#
#             if analyze_convergence:
#                 opt_name = 'CG' if self.params.get('CG_optimizer', True) else 'GD'
#                 for val_name, values in zip(['loss', 'gradient'], [self.joint_optimizer.losses, self.joint_optimizer.gradient_mags]):
#                     val_str = ' '.join(['{:.8e}'.format(v.item()) for v in values])
#                     file_name = '{}_{}.txt'.format(opt_name, val_name)
#                     with open(file_name, 'a') as f:
#                         f.write(val_str + '\n')
#                 raise RuntimeError('Exiting')
#
#         # Re-project samples with the new projection matrix
#         compressed_samples = self.project_sample(self.init_training_samples, self.projection_matrix)
#         for train_samp, init_samp in zip(self.training_samples, compressed_samples):
#             train_samp[:init_samp.shape[0],...] = init_samp
#
#         self.hinge_mask = None
#
#         # Initialize optimizer
#         self.conv_problem = ConvProblem(self.training_samples, self.y, self.filter_reg, self.sample_weights, self.response_activation)
#
#         if optimizer == 'GaussNewtonCG':
#             self.filter_optimizer = ConjugateGradient(self.conv_problem, self.filter, fletcher_reeves=self.params.fletcher_reeves,
#                                                       direction_forget_factor=self.params.direction_forget_factor, debug=(self.params.debug>=1),
#                                                       plotting=(self.params.debug>=3), visdom=self.visdom)
#         elif optimizer == 'GradientDescentL2':
#             self.filter_optimizer = GradientDescentL2(self.conv_problem, self.filter, self.params.optimizer_step_length,
#                                                       self.params.optimizer_momentum, debug=(self.params.debug >= 1),
#                                                       plotting=(self.params.debug>=3), visdom=self.visdom)
#
#         # Transfer losses from previous optimization
#         if self.params.update_projection_matrix:
#             self.filter_optimizer.residuals = self.joint_optimizer.residuals
#             self.filter_optimizer.losses = self.joint_optimizer.losses
#
#         if not self.params.update_projection_matrix:
#             self.filter_optimizer.run(self.params.init_CG_iter)
#
#         # Post optimization
#         self.filter_optimizer.run(self.params.post_init_CG_iter)
#
#         # Free memory
#         del self.init_training_samples
#         if self.params.use_projection_matrix:
#             del self.joint_problem, self.joint_optimizer
#
#     def generate_init_samples(self, im: torch.Tensor) -> TensorList:
#         """Generate augmented initial samples."""
#
#         # Compute augmentation size
#         aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
#         aug_expansion_sz = self.img_sample_sz.clone()
#         aug_output_sz = None
#         if aug_expansion_factor is not None and aug_expansion_factor != 1:
#             aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
#             aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
#             aug_expansion_sz = aug_expansion_sz.float()
#             aug_output_sz = self.img_sample_sz.long().tolist()
#
#         # Random shift operator
#         get_rand_shift = lambda: None
#         random_shift_factor = self.params.get('random_shift_factor', 0)
#         if random_shift_factor > 0:
#             get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor).long().tolist()
#
#         # Create transofmations
#         self.transforms = [augmentation.Identity(aug_output_sz)]
#         if 'shift' in self.params.augmentation:
#             self.transforms.extend([augmentation.Translation(shift, aug_output_sz) for shift in self.params.augmentation['shift']])
#         if 'relativeshift' in self.params.augmentation:
#             get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
#             self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in self.params.augmentation['relativeshift']])
#         if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
#             self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
#         if 'blur' in self.params.augmentation:
#             self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in self.params.augmentation['blur']])
#         if 'scale' in self.params.augmentation:
#             self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in self.params.augmentation['scale']])
#         if 'rotate' in self.params.augmentation:
#             self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in self.params.augmentation['rotate']])
#
#         # Generate initial samples
#         init_samples = self.params.features.extract_transformed(im, self.pos, self.target_scale, aug_expansion_sz, self.transforms)
#
#         # Remove augmented samples for those that shall not have
#         for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
#             if not use_aug:
#                 init_samples[i] = init_samples[i][0:1, ...]
#
#         # Add dropout samples
#         if 'dropout' in self.params.augmentation:
#             num, prob = self.params.augmentation['dropout']
#             self.transforms.extend(self.transforms[:1]*num)
#             for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
#                 if use_aug:
#                     init_samples[i] = torch.cat([init_samples[i], F.dropout2d(init_samples[i][0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])
#
#         return init_samples
#
#     def preprocess_sample(self, x: TensorList) -> (TensorList, TensorList):
#         if self.params.get('_feature_window', False):
#             x = x * self.feature_window
#         return x
#
#     def init_label_function(self, train_x):
#         # Allocate label function
#         self.y = TensorList([x.new_zeros(self.params.sample_memory_size, 1, x.shape[2], x.shape[3]) for x in train_x])
#
#         # Output sigma factor
#         output_sigma_factor = self.fparams.attribute('output_sigma_factor')
#         self.sigma = (self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)
#
#         # Center pos in normalized coords
#         target_center_norm = (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)
#
#         # Generate label functions
#         for y, sig, sz, ksz, x in zip(self.y, self.sigma, self.feature_sz, self.kernel_size, train_x):
#             center_pos = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
#             for i, T in enumerate(self.transforms[:x.shape[0]]):
#                 sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * sz
#                 y[i, 0, ...] = dcf.label_function_spatial(sz, sig, sample_center)
#
#         # Return only the ones to use for initial training
#         return TensorList([y[:x.shape[0], ...] for y, x in zip(self.y, train_x)])
#
#     def init_memory(self, train_x):
#         # Initialize first-frame training samples
#         self.num_init_samples = train_x.size(0)
#         self.init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])
#         self.init_training_samples = train_x
#
#         # Sample counters and weights
#         self.num_stored_samples = self.num_init_samples.copy()
#         self.previous_replace_ind = [None] * len(self.num_stored_samples)
#         self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
#         for sw, init_sw, num in zip(self.sample_weights, self.init_sample_weights, self.num_init_samples):
#             sw[:num] = init_sw
#
#         # Initialize memory
#         self.training_samples = TensorList(
#             [x.new_zeros(self.params.sample_memory_size, cdim, x.shape[2], x.shape[3]) for x, cdim in
#              zip(train_x, self.compressed_dim)])
#
#     def localize_target(self, scores_raw):
#         # Weighted sum (if multiple features) with interpolation in fourier domain
#         weight = self.fparams.attribute('translation_weight', 1.0)
#         scores_raw = weight * scores_raw
#         sf_weighted = fourier.cfft2(scores_raw) / (scores_raw.size(2) * scores_raw.size(3))
#         for i, (sz, ksz) in enumerate(zip(self.feature_sz, self.kernel_size)):
#             sf_weighted[i] = fourier.shift_fs(sf_weighted[i], math.pi * (1 - torch.Tensor([ksz[0]%2, ksz[1]%2]) / sz))
#
#         scores_fs = fourier.sum_fs(sf_weighted)
#         scores = fourier.sample_fs(scores_fs, self.output_sz)
#
#         if self.output_window is not None and not self.params.get('perform_hn_without_windowing', False):
#             scores *= self.output_window
#
#         if self.params.get('advanced_localization', False):
#             return self.localize_advanced(scores)
#
#         # Get maximum
#         max_score, max_disp = dcf.max2d(scores)
#         _, scale_ind = torch.max(max_score, dim=0)
#         max_disp = max_disp.float().cpu()
#
#         # Convert to displacements in the base scale
#         disp = (max_disp + self.output_sz / 2) % self.output_sz - self.output_sz / 2
#
#         # Compute translation vector and scale change factor
#         translation_vec = disp[scale_ind, ...].view(-1) * (self.img_support_sz / self.output_sz) * self.target_scale
#         translation_vec *= self.params.scale_factors[scale_ind]
#
#         # Shift the score output for visualization purposes
#         if self.params.debug >= 2:
#             sz = scores.shape[-2:]
#             scores = torch.cat([scores[...,sz[0]//2:,:], scores[...,:sz[0]//2,:]], -2)
#             scores = torch.cat([scores[...,:,sz[1]//2:], scores[...,:,:sz[1]//2]], -1)
#
#         return translation_vec, scale_ind, scores, None
#
#     def apply_filter(self, sample_x: TensorList):
#         return operation.conv2d(sample_x, self.filter, mode='same')
#
#     def extract_processed_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> (TensorList, TensorList):
#         x = self.extract_sample(im, pos, scales, sz)
#         return self.preprocess_sample(self.project_sample(x))
#
#     def extract_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
#         return self.params.features.extract(im, pos, scales, sz)[0]
#
#     def init_learning(self):
#         # Get window function
#         self.feature_window = TensorList([dcf.hann2d(sz).to(self.params.device) for sz in self.feature_sz])
#
#         # Filter regularization
#         self.filter_reg = self.fparams.attribute('filter_reg')
#
#         # Activation function after the projection matrix (phi_1 in the paper)
#         projection_activation = self.params.get('projection_activation', 'none')
#         if isinstance(projection_activation, tuple):
#             projection_activation, act_param = projection_activation
#
#         if projection_activation == 'none':
#             self.projection_activation = lambda x: x
#         elif projection_activation == 'relu':
#             self.projection_activation = torch.nn.ReLU(inplace=True)
#         elif projection_activation == 'elu':
#             self.projection_activation = torch.nn.ELU(inplace=True)
#         elif projection_activation == 'mlu':
#             self.projection_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
#         else:
#             raise ValueError('Unknown activation')
#
#         # Activation function after the output scores (phi_2 in the paper)
#         response_activation = self.params.get('response_activation', 'none')
#         if isinstance(response_activation, tuple):
#             response_activation, act_param = response_activation
#
#         if response_activation == 'none':
#             self.response_activation = lambda x: x
#         elif response_activation == 'relu':
#             self.response_activation = torch.nn.ReLU(inplace=True)
#         elif response_activation == 'elu':
#             self.response_activation = torch.nn.ELU(inplace=True)
#         elif response_activation == 'mlu':
#             self.response_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
#         else:
#             raise ValueError('Unknown activation')
#
#     def get_parameters(self):
#         """Get parameters."""
#         # param_module = importlib.import_module('trackers.pytracking.parameter.{}.{}'.format('atom', 'default_vot.py'))
#         # params = param_module.parameters()
#         params = parameters()
#         return params
#
#     def initialize_features(self):
#         if not getattr(self, 'features_initialized', False):
#             self.params.features.initialize()
#         self.features_initialized = True
