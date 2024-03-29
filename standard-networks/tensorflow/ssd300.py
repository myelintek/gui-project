# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================


"""SSD300 Model Configuration.

References:
  Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
  Cheng-Yang Fu, Alexander C. Berg
  SSD: Single Shot MultiBox Detector
  arXiv:1512.02325

Ported from MLPerf reference implementation:
  https://github.com/mlperf/reference/tree/ssd/single_stage_detector/ssd

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import myelindl.core.model_lib as model
import tensorflow as tf

import constants
import ssd_constants
from cnn_util import log_fn
from models import resnet_model

BACKBONE_MODEL_SCOPE_NAME = 'resnet34_backbone'


class UserModel(model.CNNModel):
  """Single Shot Multibox Detection (SSD) model for 300x300 image datasets."""

  def __init__(self, label_num=ssd_constants.NUM_CLASSES, batch_size=32,
               learning_rate=1e-3, backbone='resnet34', params=None):
    super(UserModel, self).__init__('ssd300', model.MODEL_TYPE_OBJECT_DETECTION, 300, batch_size, learning_rate,
                                      params=params)
    # For COCO dataset, 80 categories + 1 background = 81 labels
    self.label_num = label_num

    # Currently only support ResNet-34 as backbone model
    if backbone != 'resnet34':
      raise ValueError('Invalid backbone model %s for SSD.' % backbone)

    # Number of channels and default boxes associated with the following layers:
    #   ResNet34 layer, Conv7, Conv8_2, Conv9_2, Conv10_2, Conv11_2
    self.out_chan = [256, 512, 512, 256, 256, 256]

    # Number of default boxes from layers of different scales
    #   38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
    self.num_dboxes = [4, 6, 6, 6, 4, 4]

    # TODO(haoyuzhang): in order to correctly restore in replicated mode, need
    # to create a saver for each tower before graph is finalized. Use variable
    # manager for better efficiency.
    self.backbone_savers = []

    # Collected predictions for eval stage. It maps each image id in eval
    # dataset to a dict containing the following information:
    #   source_id: raw ID of image
    #   raw_shape: raw shape of image
    #   pred_box: encoded box coordinates of prediction
    #   pred_scores: scores of classes in prediction
    self.predictions = {}

    # Global step when predictions are collected.
    self.eval_global_step = 0

    # The MLPerf reference uses a starting lr of 1e-3 at bs=32.
    self.base_lr_batch_size = 32

  def skip_final_affine_layer(self):
    return True

  def custom_l2_loss(self, fp32_params):
    # TODO(haoyuzhang): check compliance with MLPerf rules
    return tf.add_n([tf.nn.l2_loss(v) for v in fp32_params
                     if 'batchnorm' not in v.name])

  def add_backbone_model(self, cnn):
    # --------------------------------------------------------------------------
    # Resnet-34 backbone model -- modified for SSD
    # --------------------------------------------------------------------------

    # Input 300x300, output 150x150
    cnn.conv(64, 7, 7, 2, 2, mode='SAME_RESNET', use_batch_norm=True)
    cnn.mpool(3, 3, 2, 2, mode='SAME')

    resnet34_layers = [3, 4, 6, 3]
    version = 'v1'

    # ResNet-34 block group 1
    # Input 150x150, output 75x75
    for i in range(resnet34_layers[0]):
      # Last argument forces residual_block to use projection shortcut, even
      # though the numbers of input and output channels are equal
      resnet_model.residual_block(cnn, 64, 1, version)

    # ResNet-34 block group 2
    # Input 75x75, output 38x38
    for i in range(resnet34_layers[1]):
      stride = 2 if i == 0 else 1
      resnet_model.residual_block(cnn, 128, stride, version, i == 0)

    # ResNet-34 block group 3
    # This block group is modified: first layer uses stride=1 so that the image
    # size does not change in group of layers
    # Input 38x38, output 38x38
    for i in range(resnet34_layers[2]):
      # The following line is intentionally commented out to differentiate from
      # the original ResNet-34 model
      # stride = 2 if i == 0 else 1
      resnet_model.residual_block(cnn, 256, stride, version, i == 0)

    # ResNet-34 block group 4: removed final block group
    # The following 3 lines are intentially commented out to differentiate from
    # the original ResNet-34 model
    # for i in range(resnet34_layers[3]):
    #   stride = 2 if i == 0 else 1
    #   resnet_model.residual_block(cnn, 512, stride, version, i == 0)

  def add_inference(self, cnn):
    cnn.use_batch_norm = True
    cnn.batch_norm_config = {'decay': ssd_constants.BATCH_NORM_DECAY,
                             'epsilon': ssd_constants.BATCH_NORM_EPSILON,
                             'scale': True}

    with tf.variable_scope(BACKBONE_MODEL_SCOPE_NAME):
      self.add_backbone_model(cnn)

    # --------------------------------------------------------------------------
    # SSD additional layers
    # --------------------------------------------------------------------------

    def add_ssd_layer(cnn, depth, k_size, stride, mode):
      return cnn.conv(depth, k_size, k_size, stride, stride,
                      mode=mode, use_batch_norm=False,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Activations for feature maps of different layers
    self.activations = [cnn.top_layer]
    # Conv7_1, Conv7_2
    # Input 38x38, output 19x19
    add_ssd_layer(cnn, 256, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 512, 3, 2, 'same'))

    # Conv8_1, Conv8_2
    # Input 19x19, output 10x10
    add_ssd_layer(cnn, 256, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 512, 3, 2, 'same'))

    # Conv9_1, Conv9_2
    # Input 10x10, output 5x5
    add_ssd_layer(cnn, 128, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 256, 3, 2, 'same'))

    # Conv10_1, Conv10_2
    # Input 5x5, output 3x3
    add_ssd_layer(cnn, 128, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 256, 3, 1, 'valid'))

    # Conv11_1, Conv11_2
    # Input 3x3, output 1x1
    add_ssd_layer(cnn, 128, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 256, 3, 1, 'valid'))

    self.loc = []
    self.conf = []

    for nd, ac, oc in zip(self.num_dboxes, self.activations, self.out_chan):
      l = cnn.conv(nd * 4, 3, 3, 1, 1, input_layer=ac,
                   num_channels_in=oc, activation=None, use_batch_norm=False,
                   kernel_initializer=tf.contrib.layers.xavier_initializer())
      scale = l.get_shape()[-1]
      # shape = [batch_size, nd * 4, scale, scale]
      l = tf.reshape(l, [self.batch_size, nd, 4, scale, scale])
      # shape = [batch_size, nd, 4, scale, scale]
      l = tf.transpose(l, [0, 1, 3, 4, 2])
      # shape = [batch_size, nd, scale, scale, 4]
      self.loc.append(tf.reshape(l, [self.batch_size, -1, 4]))
      # shape = [batch_size, nd * scale * scale, 4]

      c = cnn.conv(nd * self.label_num, 3, 3, 1, 1, input_layer=ac,
                   num_channels_in=oc, activation=None, use_batch_norm=False,
                   kernel_initializer=tf.contrib.layers.xavier_initializer())
      # shape = [batch_size, nd * label_num, scale, scale]
      c = tf.reshape(c, [self.batch_size, nd, self.label_num, scale, scale])
      # shape = [batch_size, nd, label_num, scale, scale]
      c = tf.transpose(c, [0, 1, 3, 4, 2])
      # shape = [batch_size, nd, scale, scale, label_num]
      self.conf.append(tf.reshape(c, [self.batch_size, -1, self.label_num]))
      # shape = [batch_size, nd * scale * scale, label_num]

    # Shape of locs: [batch_size, NUM_SSD_BOXES, 4]
    # Shape of confs: [batch_size, NUM_SSD_BOXES, label_num]
    locs, confs = tf.concat(self.loc, 1), tf.concat(self.conf, 1)

    # Pack location and confidence outputs into a single output layer
    # Shape of logits: [batch_size, NUM_SSD_BOXES, 4+label_num]
    logits = tf.concat([locs, confs], 2)

    cnn.top_layer = logits
    cnn.top_size = 4 + self.label_num

    return cnn.top_layer

  def get_learning_rate(self, global_step, batch_size):
    rescaled_lr = self.get_scaled_base_learning_rate(batch_size)
    # Defined in MLPerf reference model
    boundaries = [160000, 200000]
    boundaries = [b * self.base_lr_batch_size // batch_size for b in boundaries]
    decays = [1, 0.1, 0.01]
    learning_rates = [rescaled_lr * d for d in decays]
    lr = tf.train.piecewise_constant(global_step, boundaries, learning_rates)
    warmup_steps = int(118287 / batch_size * 5)
    warmup_lr = (
        rescaled_lr * tf.cast(global_step, tf.float32) / tf.cast(
            warmup_steps, tf.float32))
    return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

  def get_scaled_base_learning_rate(self, batch_size):
    """Calculates base learning rate for creating lr schedule.

    In replicated mode, gradients are summed rather than averaged which, with
    the sgd and momentum optimizers, increases the effective learning rate by
    lr * num_gpus. Dividing the base lr by num_gpus negates the increase.

    Args:
      batch_size: Total batch-size.

    Returns:
      Base learning rate to use to create lr schedule.
    """
    base_lr = self.learning_rate
    if self.params.variable_update == 'replicated':
      base_lr = self.learning_rate / self.params.num_gpus
    scaled_lr = base_lr * (batch_size / self.base_lr_batch_size)
    return scaled_lr

  def _collect_backbone_vars(self):
    backbone_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='.*'+ BACKBONE_MODEL_SCOPE_NAME)
    var_list = {}

    # Assume variables in the checkpoint are following the naming convention of
    # a model checkpoint trained with TF official model
    # TODO(haoyuzhang): the following variable name parsing is hacky and easy
    # to break if there is change in naming convention of either benchmarks or
    # official models.
    for v in backbone_vars:
      # conv2d variable example (model <-- checkpoint):
      #   v/cg/conv24/conv2d/kernel:0 <-- conv2d_24/kernel
      if 'conv2d' in v.name:
        re_match = re.search(r'conv(\d+)/conv2d/(.+):', v.name)
        if re_match:
          layer_id = int(re_match.group(1))
          param_name = re_match.group(2)
          vname_in_ckpt = self._var_name_in_official_model_ckpt(
              'conv2d', layer_id, param_name)
          var_list[vname_in_ckpt] = v

      # batchnorm varariable example:
      #   v/cg/conv24/batchnorm25/gamma:0 <-- batch_normalization_25/gamma
      elif 'batchnorm' in v.name:
        re_match = re.search(r'batchnorm(\d+)/(.+):', v.name)
        if re_match:
          layer_id = int(re_match.group(1))
          param_name = re_match.group(2)
          vname_in_ckpt = self._var_name_in_official_model_ckpt(
              'batch_normalization', layer_id, param_name)
          var_list[vname_in_ckpt] = v

    return var_list

  def _var_name_in_official_model_ckpt(self, layer_name, layer_id, param_name):
    """Return variable names according to convention in TF official models."""
    vname_in_ckpt = layer_name
    if layer_id > 0:
      vname_in_ckpt += '_' + str(layer_id)
    vname_in_ckpt += '/' + param_name
    return vname_in_ckpt

  def loss_function(self, inputs, build_network_result):
    logits = build_network_result.logits

    # Unpack model output back to locations and confidence scores of predictions
    # Shape of pred_loc: [batch_size, NUM_SSD_BOXES, 4]
    # Shape of pred_label: [batch_size, NUM_SSD_BOXES, label_num]
    pred_loc, pred_label = tf.split(logits, [4, self.label_num], 2)

    # Shape of gt_loc: [batch_size, NUM_SSD_BOXES, 4]
    # Shape of gt_label: [batch_size, NUM_SSD_BOXES, 1]
    # Shape of num_gt: [batch_size]
    _, gt_loc, gt_label, num_gt = inputs
    gt_label = tf.cast(gt_label, tf.int32)

    box_loss = self._localization_loss(pred_loc, gt_loc, gt_label, num_gt)
    class_loss = self._classification_loss(pred_label, gt_label, num_gt)

    tf.summary.scalar('box_loss', tf.reduce_mean(box_loss))
    tf.summary.scalar('class_loss', tf.reduce_mean(class_loss))
    return class_loss + box_loss

  def _localization_loss(self, pred_loc, gt_loc, gt_label, num_matched_boxes):
    """Computes the localization loss.

    Computes the localization loss using smooth l1 loss.
    Args:
      pred_loc: a flatten tensor that includes all predicted locations. The
        shape is [batch_size, num_anchors, 4].
      gt_loc: a tensor representing box regression targets in
        [batch_size, num_anchors, 4].
      gt_label: a tensor that represents the classification groundtruth targets.
        The shape is [batch_size, num_anchors, 1].
      num_matched_boxes: the number of anchors that are matched to a groundtruth
        targets, used as the loss normalizater. The shape is [batch_size].
    Returns:
      box_loss: a float32 representing total box regression loss.
    """
    mask = tf.greater(tf.squeeze(gt_label), 0)
    float_mask = tf.cast(mask, tf.float32)

    smooth_l1 = tf.reduce_sum(tf.losses.huber_loss(
        gt_loc, pred_loc,
        reduction=tf.losses.Reduction.NONE
    ), axis=2)
    smooth_l1 = tf.multiply(smooth_l1, float_mask)
    box_loss = tf.reduce_sum(smooth_l1, axis=1)

    return tf.reduce_mean(box_loss / num_matched_boxes)

  def _classification_loss(self, pred_label, gt_label, num_matched_boxes):
    """Computes the classification loss.

    Computes the classification loss with hard negative mining.
    Args:
      pred_label: a flatten tensor that includes all predicted class. The shape
        is [batch_size, num_anchors, num_classes].
      gt_label: a tensor that represents the classification groundtruth targets.
        The shape is [batch_size, num_anchors, 1].
      num_matched_boxes: the number of anchors that are matched to a groundtruth
        targets. This is used as the loss normalizater.
    Returns:
      box_loss: a float32 representing total box regression loss.
    """
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        gt_label, pred_label, reduction=tf.losses.Reduction.NONE)

    mask = tf.greater(tf.squeeze(gt_label), 0)
    float_mask = tf.cast(mask, tf.float32)

    # Hard example mining
    neg_masked_cross_entropy = cross_entropy * (1 - float_mask)
    relative_position = tf.contrib.framework.argsort(
        tf.contrib.framework.argsort(
            neg_masked_cross_entropy, direction='DESCENDING'))
    num_neg_boxes = tf.minimum(
        tf.to_int32(num_matched_boxes) * ssd_constants.NEGS_PER_POSITIVE,
        ssd_constants.NUM_SSD_BOXES)
    top_k_neg_mask = tf.cast(tf.less(
        relative_position,
        tf.tile(num_neg_boxes[:, tf.newaxis], (1, ssd_constants.NUM_SSD_BOXES))
    ), tf.float32)

    class_loss = tf.reduce_sum(
        tf.multiply(cross_entropy, float_mask + top_k_neg_mask), axis=1)

    return tf.reduce_mean(class_loss / num_matched_boxes)

  def add_backbone_saver(self):
    # Create saver with mapping from variable names in checkpoint of backbone
    # model to variables in SSD model
    backbone_var_list = self._collect_backbone_vars()
    self.backbone_savers.append(tf.train.Saver(backbone_var_list))

  def load_backbone_model(self, sess, backbone_model_path):
    for saver in self.backbone_savers:
      saver.restore(sess, backbone_model_path)

  def get_input_data_types(self, subset):
    if subset == 'validation':
      return [self.data_type, tf.float32, tf.float32, tf.float32, tf.int32]
    return [self.data_type, tf.float32, tf.float32, tf.float32]

  def get_input_shapes(self, subset):
    """Return encoded tensor shapes for train and eval data respectively."""
    if subset == 'validation':
      # Validation data shapes:
      # 1. images
      # 2. ground truth locations of boxes
      # 3. ground truth classes of objects in boxes
      # 4. source image IDs
      # 5. raw image shapes
      return [
          [self.batch_size, self.image_size, self.image_size, self.depth],
          [self.batch_size, ssd_constants.MAX_NUM_EVAL_BOXES, 4],
          [self.batch_size, ssd_constants.MAX_NUM_EVAL_BOXES, 1],
          [self.batch_size],
          [self.batch_size, 3],
      ]

    # Training data shapes:
    # 1. images
    # 2. ground truth locations of boxes
    # 3. ground truth classes of objects in boxes
    # 4. numbers of objects in images
    return [
        [self.batch_size, self.image_size, self.image_size, self.depth],
        [self.batch_size, ssd_constants.NUM_SSD_BOXES, 4],
        [self.batch_size, ssd_constants.NUM_SSD_BOXES, 1],
        [self.batch_size]
    ]

  def accuracy_function(self, inputs, logits):
    """Returns the ops to measure the mean precision of the model."""
    try:
      import ssd_dataloader  # pylint: disable=g-import-not-at-top
      from object_detection.box_coders import faster_rcnn_box_coder  # pylint: disable=g-import-not-at-top
      from object_detection.core import box_coder  # pylint: disable=g-import-not-at-top
      from object_detection.core import box_list  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise ImportError('To use the COCO dataset, you must clone the '
                        'repo https://github.com/tensorflow/models and add '
                        'tensorflow/models and tensorflow/models/research to '
                        'the PYTHONPATH, and compile the protobufs by '
                        'following https://github.com/tensorflow/models/blob/'
                        'master/research/object_detection/g3doc/installation.md'
                        '#protobuf-compilation ; To evaluate using COCO'
                        'metric, download and install Python COCO API from'
                        'https://github.com/cocodataset/cocoapi')

    # Unpack model output back to locations and confidence scores of predictions
    # pred_locs: relative locations (coordiates) of objects in all SSD boxes
    # shape: [batch_size, NUM_SSD_BOXES, 4]
    # pred_labels: confidence scores of objects being of all categories
    # shape: [batch_size, NUM_SSD_BOXES, label_num]
    pred_locs, pred_labels = tf.split(logits, [4, self.label_num], 2)

    ssd_box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=ssd_constants.BOX_CODER_SCALES)
    anchors = box_list.BoxList(
        tf.convert_to_tensor(ssd_dataloader.DefaultBoxes()('ltrb')))
    pred_boxes = box_coder.batch_decode(
        encoded_boxes=pred_locs, box_coder=ssd_box_coder, anchors=anchors)

    pred_scores = tf.nn.softmax(pred_labels, axis=2)

    # TODO(haoyuzhang): maybe use `gt_boxes` and `gt_classes` for visualization.
    if len(inputs) > 4:
      _, gt_boxes, gt_classes, source_id, raw_shape = inputs  # pylint: disable=unused-variable 
      return {
            (constants.UNREDUCED_ACCURACY_OP_PREFIX +
             ssd_constants.PRED_BOXES): pred_boxes,
            (constants.UNREDUCED_ACCURACY_OP_PREFIX +
             ssd_constants.PRED_SCORES): pred_scores,
            # TODO(haoyuzhang): maybe use these values for visualization.
            # constants.UNREDUCED_ACCURACY_OP_PREFIX+'gt_boxes': gt_boxes,
            # constants.UNREDUCED_ACCURACY_OP_PREFIX+'gt_classes': gt_classes,
            (constants.UNREDUCED_ACCURACY_OP_PREFIX +
             ssd_constants.SOURCE_ID): source_id,
            (constants.UNREDUCED_ACCURACY_OP_PREFIX +
             ssd_constants.RAW_SHAPE): raw_shape
        }
    else:
      _, gt_boxes, gt_classes, source_id = inputs  # pylint: disable=unused-variable

      return {
            (constants.UNREDUCED_ACCURACY_OP_PREFIX +
             ssd_constants.PRED_BOXES): pred_boxes,
            (constants.UNREDUCED_ACCURACY_OP_PREFIX +
             ssd_constants.PRED_SCORES): pred_scores,
            # TODO(haoyuzhang): maybe use these values for visualization.
            # constants.UNREDUCED_ACCURACY_OP_PREFIX+'gt_boxes': gt_boxes,
            # constants.UNREDUCED_ACCURACY_OP_PREFIX+'gt_classes': gt_classes,
            (constants.UNREDUCED_ACCURACY_OP_PREFIX +
             ssd_constants.SOURCE_ID): source_id,
        }

  def postprocess(self, results):
    """Postprocess results returned from model."""
    try:
      import coco_metric  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise ImportError('To use the COCO dataset, you must clone the '
                        'repo https://github.com/tensorflow/models and add '
                        'tensorflow/models and tensorflow/models/research to '
                        'the PYTHONPATH, and compile the protobufs by '
                        'following https://github.com/tensorflow/models/blob/'
                        'master/research/object_detection/g3doc/installation.md'
                        '#protobuf-compilation ; To evaluate using COCO'
                        'metric, download and install Python COCO API from'
                        'https://github.com/cocodataset/cocoapi')

    pred_boxes = results[ssd_constants.PRED_BOXES]
    pred_scores = results[ssd_constants.PRED_SCORES]
    # TODO(haoyuzhang): maybe use these values for visualization.
    # gt_boxes = results['gt_boxes']
    # gt_classes = results['gt_classes']
    source_id = results[ssd_constants.SOURCE_ID]
    raw_shape = results[ssd_constants.RAW_SHAPE]

    # COCO evaluation requires processing COCO_NUM_VAL_IMAGES exactly once. Due
    # to rounding errors (i.e., COCO_NUM_VAL_IMAGES % batch_size != 0), setting
    # `num_eval_epochs` to 1 is not enough and will often miss some images. We
    # expect user to set `num_eval_epochs` to >1, which will leave some unused
    # images from previous steps in `predictions`. Here we check if we are doing
    # eval at a new global step.
    if results['global_step'] > self.eval_global_step:
      self.eval_global_step = results['global_step']
      self.predictions.clear()

    for i in range(self.get_batch_size()):
      self.predictions[int(source_id[i])] = {
          ssd_constants.PRED_BOXES: pred_boxes[i],
          ssd_constants.PRED_SCORES: pred_scores[i],
          ssd_constants.SOURCE_ID: source_id[i],
          ssd_constants.RAW_SHAPE: raw_shape[i]
      }

    # COCO metric calculates mAP only after a full epoch of evaluation. Return
    # dummy results for top_N_accuracy to be compatible with benchmar_cnn.py.
    if len(self.predictions) >= ssd_constants.COCO_NUM_VAL_IMAGES:
      log_fn('Got results for all {:d} eval examples. Calculate mAP...'.format(
          ssd_constants.COCO_NUM_VAL_IMAGES))
      annotation_file = os.path.join(self.params.data_dir,
                                     ssd_constants.ANNOTATION_FILE)
      eval_results = coco_metric.compute_map(self.predictions.values(),
                                             annotation_file)
      self.predictions.clear()
      ret = {'top_1_accuracy': 0., 'top_5_accuracy': 0.}
      for metric_key, metric_value in eval_results.items():
        ret[constants.SIMPLE_VALUE_RESULT_PREFIX + metric_key] = metric_value
      return ret
    log_fn('Got {:d} out of {:d} eval examples.'
           ' Waiting for the remaining to calculate mAP...'.format(
               len(self.predictions), ssd_constants.COCO_NUM_VAL_IMAGES))
    return {'top_1_accuracy': 0., 'top_5_accuracy': 0.}

  def get_synthetic_inputs(self, input_name, nclass):
    """Generating synthetic data matching real data shape and type."""
    inputs = tf.random_uniform(
        self.get_input_shapes('train')[0], dtype=self.data_type)
    inputs = tf.contrib.framework.local_variable(inputs, name=input_name)
    boxes = tf.random_uniform(
        [self.batch_size, ssd_constants.NUM_SSD_BOXES, 4], dtype=tf.float32)
    classes = tf.random_uniform(
        [self.batch_size, ssd_constants.NUM_SSD_BOXES, 1], dtype=tf.float32)
    nboxes = tf.random_uniform(
        [self.batch_size], minval=1, maxval=10, dtype=tf.float32)
    return (inputs, boxes, classes, nboxes)
