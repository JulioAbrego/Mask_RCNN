"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import glob
import random
import math
import datetime
import itertools
import json
import multiprocessing
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import scipy.io
import skimage.io

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

import common



ROOT_DIR = "/deepmatter"
MODEL_DIR = os.path.join(ROOT_DIR, "models/mask_rcnn/")


############################################################
#  Utility Functions
############################################################

# TODO: use this instead of tensor_summary() and print()
def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


# TODO: Verify if layer_params is really needed
def layer_params():
    return {}

def batch_norm_params():
    return {
        "momentum": 0.99,  # as recommended by the Batch Re-Norm paper TODO: affected by GPU_COUNT
        "epsilon": 1e-5,
# xxxx        "renorm_warmup": 10000,
    }

# Batch Normaliztion

class BatchNorm(KL.BatchNormalization):
    """Disable batch normaliztion.
    TODO: elaborate
    """
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)



############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=False):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias, **layer_params())(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a', **batch_norm_params())(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size),
               padding='same', name=conv_name_base + '2b', use_bias=use_bias, **layer_params())(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b', **batch_norm_params())(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias, **layer_params())(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c', **batch_norm_params())(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res'+str(stage)+block+'_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, 
               strides=(2, 2), use_bias=False):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=use_bias, **layer_params())(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a', **batch_norm_params())(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias, **layer_params())(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b', **batch_norm_params())(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias, **layer_params())(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c', **batch_norm_params())(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias, **layer_params())(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1', 
                                    **batch_norm_params())(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res'+str(stage)+block+'_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False, **layer_params())(x)
    x = BatchNorm(axis=3, name='bn_conv1', **batch_norm_params())(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98+i))
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', use_bias=False)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', use_bias=False)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', use_bias=False)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


# Proposal Layer

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    return clipped


class ProposalLayer(KE.Layer):
    """Filters and selects from anchors to generate detection proposals.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns: TODO:
    [batch, num_rois, 4]. Each row is: y1, x1, y2, x2
    """

    def __init__(self, proposal_count, nms_threshold, anchors, config=None, **kwargs):
        """
        anchors: [N, (y1, x1, y2, x2)] anchors defined in image coordinates
        """
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.anchors = anchors.astype(np.float32) # xxxx

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:,:,1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Base anchors
        anchors = self.anchors

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(10000, self.anchors.shape[0])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        scores = common.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        deltas = common.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        anchors = common.batch_slice(ix, lambda x: tf.gather(anchors, x), self.config.IMAGES_PER_GPU, 
                              names=["pre_nms_anchors"])
        
        # Clip anchors to image. TODO: this might not be correct.
        # Result: [N, (y1, x1, y2, x2)]
        # TODO: Clip to window inside the image?
#         height, width = self.config.IMAGE_SHAPE[:2]
#         window = np.array([0, 0, height, width]).astype(np.float32)
#         anchors = clip_boxes_graph(tf.cast(self.anchors, tf.float32), window)

        # Apply deltas to anchors to get the refined anchors. [batch, N, (y1, x1, y2, x2)]
        boxes = common.batch_slice([anchors, deltas],
                                   lambda x, y: apply_box_deltas_graph(x, y),
                                   self.config.IMAGES_PER_GPU, 
                                   names=["refined_anchors"])

        # Clip to image. [batch, N, (y1, x1, y2, x2)]
        # TODO: Clip to window inside the image?
        height, width = self.config.IMAGE_SHAPE[:2]
        window = np.array([0, 0, height, width]).astype(np.float32)
        boxes = common.batch_slice(boxes, 
                                   lambda x: clip_boxes_graph(x, window),
                                   self.config.IMAGES_PER_GPU, 
                                   names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces quality for small objects.

        # Filter out boxes that don't overlap the image at all.
        # xxx needed?
#         y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
#         area = (x2 - x1) * (y2 - y1)
#         indices = tf.where(tf.greater(area, 0.0))[:,0]
#         boxes = tf.gather(boxes, indices)
#         scores = tf.gather(scores, indices)

        # Normalize dimensions to range of 0 to 1.
        normalized_boxes = boxes / np.array([[height, width, height, width]])

        # Non-max suppression
        def nms(normalized_boxes, scores):
            indices = tf.image.non_max_suppression(normalized_boxes, scores, 
                                                   self.proposal_count, 
                                                   self.nms_threshold, 
                                                   name="rpn_non_max_suppression")
            proposals = tf.gather(normalized_boxes, indices)
            # Pad if needed
            padding = self.proposal_count - tf.shape(proposals)[0]
            proposals = tf.concat([proposals, tf.zeros([padding, 4])], 0)
            return proposals
        proposals = common.batch_slice([normalized_boxes, scores], nms, self.config.IMAGES_PER_GPU)

        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


# ROIAlign Layer

def log2(x):
    """Implement Log2 because TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """
    A pyramid version of ROIAlign layer that aggregates ROIs from different
    levels of a feature pyramid.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7] todo: right?
    - image_shape: [height, width, chanells]. Shape of input image

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    def __init__(self, pool_shape, image_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_shape = tuple(image_shape)

    def call(self, inputs):
        # Crop boxes. [batch, num_boxes, (y1, x1, y2, x2)] in normalized coordinates
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(self.image_shape[0] * self.image_shape[1], tf.float32)
        roi_level = log2(tf.sqrt(h*w) / (224.0/tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals. Otherwise, it causes
            # an odd bug in multi-GPU training (complains that crop_and_resize
            # receives box indicies outside the range, which is not accurate).
            # TODO: the above bug still happens despite this in TF 1.2
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize.
            # Result shape: [batch * num_boxes, pool_height, pool_width, channels]
            #
            # From Mask R-CNN paper: We sample four regular locations, so that we 
            # can evaluate either max or average pooling. In fact, interpolating 
            # only a single value at each bin center (without pooling) is nearly 
            # as effective.
            #
            # We use the simplified approach here of a single value per bin, which
            # tf.crop_and_resize() does naturally.
            pooled.append(tf.image.crop_and_resize(feature_maps[i], level_boxes, 
                                                   box_indices, self.pool_shape, 
                                                   method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange the pooled features to match the order of the original boxes.
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:,2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1], )



# Refinement Layer

def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes


def refine_rois(rois, probs, deltas, window, config):
    """Refine the ROIs of one image  
    TODO

    Inputs:
    rois: [N, (y1, x1, y2, x2)] in normalized coordinates
    probs: [N, num_classes]. Class probabilities.
    deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific 
            bounding box deltas. 
    window: (y1, x1, y2, x2) in image coordinates. The part of the image
        that contains the image excluding the padding.
    """
    # Class IDs per ROI
    class_ids = np.argmax(probs, axis=1)
    # Class probability of the top class of each ROI
    class_scores = probs[np.arange(class_ids.shape[0]), class_ids]
    # Class-specific bounding box deltas
    deltas_specific = deltas[np.arange(deltas.shape[0]), class_ids]
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = common.apply_box_deltas(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Convert coordiates to image domain
    # todo: probably better to keep them normalized until later
    height, width = config.IMAGE_SHAPE[:2]
    refined_rois *= np.array([height, width, height, width])
    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)
    # Round and cast to int since we're deadling with pixels now
    refined_rois = np.rint(refined_rois).astype(np.int32)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = np.where(class_ids > 0)[0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep = np.intersect1d(
            keep, np.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[0])

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_rois = refined_rois[keep]
    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = common.non_max_suppression(
            pre_nms_rois[ixs], pre_nms_scores[ixs],
            config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)

    # Keep top ROIs
    roi_count = 100  # todo: move to config
    top_ids = np.argsort(class_scores[keep])[::-1][:roi_count]
    keep = keep[top_ids]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = np.hstack((refined_rois[keep],
                        class_ids[keep][..., np.newaxis], 
                        class_scores[keep][..., np.newaxis]))
    return result


############################################################
#  Proposal Target Layer
############################################################

def trim_zeros_graph(boxes):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    
    boxes: [N, 4] matrix of boxes.
    """
    area = tf.boolean_mask(boxes, tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool))
    
    

def proposal_targets_graph(rpn_rois, gt_boxes, gt_masks, config):
    """
    Inputs:
        rpn_rois: [num_rois, (y1, x1, y2, x2)] in normalized coordinates. Padded with 
            zeros if ROIs are less than the array size.
        gt_boxes: [instance count, (y1, x1, y2, x2, class_id)] in normalized coordinates
        gt_masks: [height, width, instance count] TODO: what size?
    
    Returns:
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates.
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw), class_id)]. 
                Rows are class-specific bbox refinments.
        masks: [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox boundaries
            and resized to neural network output size.
        TODO: class specific not done yet
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(rpn_rois)[0], 0), [rpn_rois], name="rpn_roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        rpn_rois = tf.identity(rpn_rois)
        
    # Trim ROIs to only ones with real values
    rpn_rois = tf.boolean_mask(rpn_rois, tf.cast(tf.reduce_sum(tf.abs(rpn_rois), axis=1), tf.bool))
    
    # Compute overlaps matrix [rpn_rois, gt_boxes]
    # 1. Tile GT boxes and repeate ROIs tensor. This
    # allows us to compare every ROI against every GT box without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it 
    # using tf.tile() and tf.reshape.
    rois = tf.reshape(tf.tile(tf.expand_dims(rpn_rois, 1), 
                              [1, 1, tf.shape(gt_boxes)[0]]), [-1, 4])
    boxes = tf.tile(gt_boxes, [tf.shape(rpn_rois)[0], 1])
    # 2. Compute intersections
    roi_y1, roi_x1, roi_y2, roi_x2 = tf.split(rois, 4, axis=1)
    box_y1, box_x1, box_y2, box_x2, class_ids = tf.split(boxes, 5, axis=1)
    y1 = tf.maximum(roi_y1, box_y1)
    x1 = tf.maximum(roi_x1, box_x1)
    y2 = tf.minimum(roi_y2, box_y2)
    x2 = tf.minimum(roi_x2, box_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    roi_area = (roi_y2 - roi_y1) * (roi_x2 - roi_x1)
    box_area = (box_y2 - box_y1) * (box_x2 - box_x1)
    union = roi_area + box_area - intersection
    # 4. Compute IoU and reshape to [rois, boxes]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(rpn_rois)[0], tf.shape(gt_boxes)[0]])
    
    # Determine postive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box. 
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:,0]
    # 2. Negative ROIs are those with < 0.5 with every GT box.
    negative_indices = tf.where(roi_iou_max < 0.5)[:,0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    # Negative ROIs. Fill the rest of the batch.
    negative_count = config.TRAIN_ROIS_PER_IMAGE - tf.shape(positive_indices)[0]
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(rpn_rois, positive_indices)
    negative_rois = tf.gather(rpn_rois, negative_indices)
    
    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    
    # Compute bbox refinement for positive ROIs
    deltas = common.box_refinement_graph(positive_rois, roi_gt_boxes[:,:4])
    deltas /= config.BBOX_STD_DEV
    # Add class_id to each delta. Used in the loss function.
#     deltas = tf.concat([deltas, roi_gt_boxes[:,4:5]], axis=1)
    
    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
    
    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI corrdinates from normalized image space 
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2, _ = tf.split(roi_gt_boxes, 5, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes, 
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)
    
    # Threshold mask pixels at 0.5. Use binary masks with 
    # binary cross entropy loss.
    masks = tf.cast(masks + 0.5, tf.int32)
    masks = tf.cast(masks, tf.float32)  # TODO: maybe this should be done in the loss function
    
    # Append negative ROIs and pad bbox deltas and masks, which
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N+P), (0, 0)])
    deltas = tf.pad(deltas, [(0, N+P), (0, 0)])
    masks = tf.pad(masks, [[0, N+P], (0, 0), (0, 0)])

    return rois, roi_gt_boxes[:,4], deltas, masks

class ProposalTargetLayer(KE.Layer):
    """
    Subsample RPN ROIs and generate box refinment, class_ids, and masks for each ROI.
    TODO
    
    Inputs:
    rpn_rois: [batch, num_rois, (y1, x1, y2, x2)] in normalized coordinates
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2, class_id)] in normalized coordinates
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type.
    
    Returns:
    TODO
    
    """
    def __init__(self, config, **kwargs):
        super(ProposalTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        # RPN ROIs are in normalized coordinates
        rpn_rois = inputs[0]
        gt_boxes = inputs[1]
        gt_masks = inputs[2]

        # Set names for readability
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = common.batch_slice(
            [rpn_rois, gt_boxes, gt_masks],
            lambda x, y, z: proposal_targets_graph(x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        
        return outputs

        
    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, 1),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 5),  # bboxes
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0], 
             self.config.MASK_SHAPE[1])
        ]
    
    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


# TODO: fix for batch_size > 1
class RefinementLayer(KE.Layer):
    """
    TODO
    
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in image coordinates.
    """
        
    def __init__(self, config=None, **kwargs):
        super(RefinementLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        def wrapper(rois, mrcnn_class, mrcnn_bbox, image_meta):
            # currently supports one image per batch
            b = 0
            _, _, window, _ = common.parse_image_meta(image_meta)
            detections = refine_rois(rois[b], mrcnn_class[b], mrcnn_bbox[b], 
                                     window[b], self.config)
            # Pad if detections < MAX_DETECTIONS to keep array shape constant.
            gap = self.config.MAX_DETECTIONS - detections.shape[0]
            assert gap >= 0
            if gap > 0:
                detections = np.pad(detections, [(0, gap), (0, 0)],
                              'constant', constant_values=0)
                
            # Cast to float32  TODO: why? track where float64 is introduced
            detections = detections.astype(np.float32)
            
            # Reshape output
            # [batch, num_detections, (y1, x1, y2, x2, class_score)] in image coordinates.
            return np.reshape(detections, [1, self.config.MAX_DETECTIONS, 6])
        
        # Return wrapped function
        return tf.py_func(wrapper, inputs, tf.float32)
        
    def compute_output_shape(self, input_shape):
        return (None, self.config.MAX_DETECTIONS, 6)

# Region Proposal Network (RPN)

def build_rpn_graph(feature_map, anchors_per_location, config):
    """
    feature_map: backbone shared features [batch, height, width, channels]
    
    Returns:
    rpn_class: [batch, height, width, 2]
    rpn_bbox: [batch, height, width, 4]
#    rpn_rois: [batch, num_rois, 6]. Rows are x1, y1, x2, y2, image_index, rpn_score
    """
    # TODO: does stride of 2 cause alignment issues if the feature map is not even?
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu', 
                      strides=config.RPN_ANCHOR_STRIDE,
                      **layer_params(), name='rpn_conv_shared')(feature_map)
    
    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear', 
                 **layer_params(),
                 name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
# todo:    rpn_class_logits = KL.Reshape([num_anchors, 2])(x)
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_class = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)
    
    # Anchor bounding box refinements. [batch, height, width, anchors per location, depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location*4, (1, 1), padding="valid", activation='linear', 
                 **layer_params(), name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
# todo:     rpn_bbox = KL.Reshape([num_anchors, 4], name="rpn_bbox_xxx")(x)
    rpn_bbox = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    
    return [rpn_class_logits, rpn_class, rpn_bbox]
    

def build_rpn_model(config, anchors_per_location, depth):
    """
    Wraps the RPN graph to allow using it multiple times with shared weights.
    """
    input_feature_map = KL.Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = build_rpn_graph(input_feature_map, anchors_per_location, config)
    return KM.Model([input_feature_map], outputs, name="rpn_model")
    


# FPN

def build_fpn_classifier_graph(rois, feature_maps, config):
    """
    TODO
    rois: Proposal boxes in normalized coordinates. [N, (y1, x1, y2, x2)].
    feature_maps: List of feature maps from diffent layers of the pyramid [P2, P3, P4, P5]
    """
    # ROI Pooling
    # Shape: [batch, num_boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([config.POOL_SIZE, config.POOL_SIZE], config.IMAGE_SHAPE,
                        name="roi_align_classifier")([rois] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(1024, (config.POOL_SIZE, config.POOL_SIZE), 
                                   padding="valid", **layer_params()),
                          name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3, **batch_norm_params()),
                          name='mrcnn_class_bn1')(x)
    x = KL.Activation('relu')(x)
    
    x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1), **layer_params()),
                               name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3, **batch_norm_params()),
                          name='mrcnn_class_bn2')(x)
    x = KL.Activation('relu')(x)
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(config.NUM_CLASSES, **layer_params()),
                                           name='mrcnn_class_logits')(shared)
    mrcnn_class = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(config.NUM_CLASSES*4, activation='linear', **layer_params()), 
                          name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], config.NUM_CLASSES, 4), name="mrcnn_bbox")(x)
    
    return mrcnn_class_logits, mrcnn_class, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, config):
    """Generate a mask for each class
    TODO
    rois: [batch, roi_count, coordinates]. coordinates=[x1, y1, x2, y1, image_index, ROI score]
    feature_maps: List of feature maps from diffent layers of the pyramid [P2, P3, P4, P5]

    Returns: Masks [batch, roi_count, height, width, num_classes]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE], config.IMAGE_SHAPE,
                        name="roi_align_mask")([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same", **layer_params()),
                          name="mrcnn_mask_conv1")(x)
    # todo: layer name should be assigned to the inner layer rather than TimeDistributed.
    x = KL.TimeDistributed(BatchNorm(axis=3, **batch_norm_params()),
                          name='mrcnn_mask_bn1')(x)
    x = KL.Activation('relu')(x)
    
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same", **layer_params()),
                          name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3, **batch_norm_params()),
                          name='mrcnn_mask_bn2')(x)
    x = KL.Activation('relu')(x)
    
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same", **layer_params()),
                          name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3, **batch_norm_params()),
                          name='mrcnn_mask_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same", **layer_params()),
                          name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3, **batch_norm_params()),
                          name='mrcnn_mask_bn4')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2,2), strides=2, activation="relu", **layer_params()), 
                          name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(config.NUM_CLASSES, (1, 1), strides=1, activation="sigmoid", **layer_params()), 
                                   name="mrcnn_mask")(x)
    return x


############################################################
#  Loss Functions
############################################################

# todo: is this the right place?
def batch_pack(x, counts, batch_size):
    """
    Picks different number of values from each row 
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(batch_size):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1-less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive, 
            -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss, 
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Crossentropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class, 
                                             output=rpn_class_logits, 
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
#     loss = tf.reshape(loss, [1, 1])  # Keep all values as tensors
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """
    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive, 
            -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
#     output_shape = rpn_bbox.get_shape()
#     output_shape_tensor = tf.shape(rpn_bbox)

    # Reshape values to a standard shape.
#     target_bbox = K.reshape(target_bbox, (-1, 4))
#     rpn_bbox = K.reshape(rpn_bbox, (-1, 4))
#     rpn_match = K.reshape(rpn_match, (-1,))
    rpn_match = K.squeeze(rpn_match, -1)
    
    # Positive anchors contribute to the loss, but negative and 
    # neutral anchors (match value of 0 or -1) don't.
    indices = tf.where(K.equal(rpn_match, 1))
    
    # Pick bbox deltas that contribute to the loss.
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack(target_bbox, batch_counts, config.IMAGES_PER_GPU)
    
    # todo: use smooth_l1
    diff = K.abs(target_bbox - rpn_bbox)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1-less_than_one) * (diff - 0.5)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
#     loss = tf.reshape(loss, [1, 1])  # Keep all values as tensors
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, class_id_mask):
    """
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    class_id_mask: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    target_class_ids = tf.cast(target_class_ids, 'int64')  # todo: why cast?
    
    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    pred_masks = tf.gather(class_id_mask[0], pred_class_ids)
    
    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes not in the class mask.
    loss = loss * pred_masks
    
    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_masks)
#     loss = tf.reshape(loss, [1, 1])  # Keep all values as tensors
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Implements Smooth-L1 loss for Mask R-CNN bounding box refinement.
    
    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))
    
    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
    
    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0, 
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
#     loss = tf.reshape(loss, [1, 1])  # Keep all values as tensors
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """
    target_masks: [batch, proposals, height, width]. 
        A float32 tensor with values of 0 or 1. TODO: why float? could be boolean
        Uses zero padding to fill the array.
    target_class_ids: [batch, num proposals]. Integer class IDs. Uses 
        zero padding at the end of the array.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)
    
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    # todo: potential for issue with Keras's crossentropy due to clipping
    loss = K.switch(tf.size(y_true) > 0, 
                    K.binary_crossentropy(target=y_true, output=y_pred), 
                    tf.constant(0.0))
    loss = K.mean(loss)
#     loss = tf.reshape(loss, [1, 1])
    return loss



############################################################
#  Training
############################################################

# Compile the model
def compile_model(model, learning_rate, momentum, config, trace=False):
    optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, 
                                     clipnorm=5.0)  # todo: confirm clip norm

    if trace:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        kwargs = {"options":run_options, 
                  "run_metadata": run_metadata}
        
        # Attach run_metadata to the model object for easy access
        model._run_metadata = run_metadata
    else:
        kwargs = {}
        
    # Add Losses
    # First, clear previously set losses to avoid duplication
    model._losses = []
    model._per_input_losses = {}
    loss_names = ["rpn_class_loss", "rpn_bbox_loss", 
                  "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
    for name in loss_names:
        layer = model.get_layer(name)
        if layer.output in model.losses:
            continue
        model.add_loss(tf.reduce_mean(layer.output, keep_dims=True))

    # Add L2 Regularization
    reg_losses = [keras.regularizers.l2(config.WEIGHT_DECAY)(w)
                  for w in model.trainable_weights]
    model.add_loss(tf.add_n(reg_losses))
        
    # Compile
    model.compile(optimizer=optimizer, loss=[None]*len(model.outputs), **kwargs)

    # Add metrics
    for name in loss_names:
        if name in model.metrics_names:
            continue
        layer = model.get_layer(name)
        model.metrics_names.append(name)
        model.metrics_tensors.append(tf.reduce_mean(layer.output, keep_dims=True))


def set_trainable(model, layer_regex, indent=0):
    """Sets model layers as trainable if their names match
    the given regular expression.
    """
    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    layers = model.inner_model.layers if hasattr(model, "inner_model") else model.layers

    for layer in layers:
        # Is the layer a model?
        if layer.__class__.__name__ == 'Model':
            print("In model: ", layer.name)
            set_trainable(layer, layer_regex, indent=indent+4)
            continue

        if not layer.weights:
            continue
        # Is it trainable?
        trainable = bool(re.fullmatch(layer_regex, layer.name))
        # Update layer. If layer is a container, update inner layer.
        if layer.__class__.__name__ == 'TimeDistributed':
            layer.layer.trainable = trainable
        else:
            layer.trainable = trainable
        # Print trainble layer names
        if trainable:
            print("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))




############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    
    augment: If true, apply random image augmentation. Currently, only 
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height 
        and width as the original image. These can be big, for example 
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the 
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    bbox: [instance_count, (y1, x1, y2, x2, class_id)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    shape = image.shape
    image, window, scale, padding = common.resize_image(
        image, 
        min_dim=config.IMAGE_MIN_DIM, 
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    mask = common.resize_mask(mask, scale, padding)

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = common.extract_bboxes(mask)
    
    # Add class_id as the last value in bbox  todo: is this good?
    bbox = np.hstack([bbox, class_ids[:,np.newaxis]])

    # Class mask 
    # Different datasets have different classes, so build a mask
    # that marks the classes supported in the dataset of this image.
    # todo: rename to active_class_ids
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[class_ids] = 1
    
    # Resize masks to reduce memory
    # todo: add more details
    if use_mini_mask:
        mask = common.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
            
    # Image info  TODO: define shape in comments
    image_meta = common.compose_image_meta(image_id, shape, window, active_class_ids)
    
    return image, image_meta, bbox, mask


def build_image_proposal_targets(rpn_rois, gt_boxes, gt_masks, config):
    """
    Inputs:
    rpn_rois: [num_rois, (y1, x1, y2, x2, image_id, score)]  # TODO: don't need image_id and score here
    gt_boxes: [instance count, (y1, x1, y2, x2, class_id)]
    gt_masks: [height, width, instance count] TODO: could be mini or full
    
    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Int class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, 5]. Rows are class-specific 
            bbox refinments [y, x, log(h), log(w), weight].
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped 
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(gt_masks.dtype)
    
    # We don't add GT Boxes to ROIs because according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_boxes[:,4] > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain at least one instance."
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:,:,instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i][:4]
        overlaps[:,i] = common.compute_iou(gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]), rpn_roi_iou_argmax]
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]  # GT box assigned to each ROI

    # Positive ROIs are those with >= 0.5 IoU with a GT box. 
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU between 0.1 and 0.5 (simple hard example mining).
    # TODO: To hard example mine or not to hard example mine? that's the question.
#     bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indicies of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.
        
        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
           "keep doesn't match ROI batch size {}, {}".format(keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep, :4]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    class_ids = roi_gt_boxes[:,4].astype(np.int32)
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox shifts. [y, x, log(h), log(w), weight]. Weight is 0 or 1 to
    # determine if a bbox is included in the loss.
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES, 5), dtype=np.float32)
    pos_ids = np.where(class_ids > 0)[0]
    bboxes[pos_ids, class_ids[pos_ids], :4] = common.compute_box_refinement(rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    bboxes[pos_ids, class_ids[pos_ids], 4] = 1  # weight = 1 to influence the loss
    # Normalize bbox refinments
    bboxes[:, :, :4] /= config.BBOX_STD_DEV

    # Generate class-specific target masks.
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES), 
                     dtype=np.float32)
    for i in pos_ids:
        class_id = class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]
        
        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id][:4]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(scipy.misc.imresize(class_mask.astype(float), (gt_h, gt_w), 
                                             interp='nearest') / 255.0).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder
            
        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i][:4].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = scipy.misc.imresize(m.astype(float), config.MASK_SHAPE, interp='nearest') / 255.0
        masks[i,:,:,class_id] = mask
        
    return rois, class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive anchors
    and the shifts needed to refine them to match their corresponding GT boxes.
    
    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2, class_id)]
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (y1, x1, y2, x2)]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))
    
    # Areas of anchors and GT boxes
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    anchor_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])

    # Compute overlaps [num_anchors, num_gt_boxes]
    # Each cell contains the IoU of an anchor and GT box.
    overlaps = np.zeros((anchors.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i][:4]
        overlaps[:,i] = common.compute_iou(gt, anchors, gt_box_area[i], anchor_area)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above, 
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. It gets overwritten if a gt box is matched to them.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[anchor_iou_max < 0.3] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: Caffe code sets multiple anchors per gt box if they have similar IoU
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1
    # 4. Set anchors that exceed image boundary as neutral.
    # TODO: Does Mask R-CNN do this?
    # TODO: if this sets an anchor that a GT box is matched with it would
    #       leave the GT box without an anchor.
#     buffer = 0  # how far outside the border is allowed
#     out_anchors = ((anchors[:,0] < -buffer) |
#                    (anchors[:,1] < -buffer) |
#                    (anchors[:,2] > image_shape[1] + buffer) |
#                    (anchors[:,3] > image_shape[0] + buffer))
#     rpn_match[out_anchors] = 0

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shifts needed to transform them 
    # to corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i], :4]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    
    return rpn_match, rpn_bbox
 
    
def generate_random_rois(image_shape, count, gt_boxes):
    """TODO
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)
    
    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i,:4]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1-h, 0)
        r_y2 = min(gt_y2+h, image_shape[0])
        r_x1 = max(gt_x1-w, 0)
        r_x2 = min(gt_x2+w, image_shape[1])
        
        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes 
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box*2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box*2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:,0] - y1y2[:,1]) >= threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:,0] - x1x2[:,1]) >= threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break
        
        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box*i:rois_per_box*(i+1)] = box_rois
    
    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes 
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:,0] - y1y2[:,1]) >= threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:,0] - x1x2[:,1]) >= threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break
    
    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def data_generator(dataset, config, shuffle=True, augment=True, random_rois=0, batch_size=1, 
                   limit_image_ids=None, roi_predictions=False):
    """
    random_rois: 
    
    TODO: 
    """
    # TODO: If the ROI doesn't fit the GT box perfectly then the mask should be shifted as well
    
    b = 0  # batch item index
    image_index = -1
    image_ids = limit_image_ids or np.copy(dataset.image_ids)
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = common.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, 
                                              config.RPN_ANCHOR_RATIOS,
                                              config.BACKBONE_SHAPES,
                                              config.BACKBONE_STRIDES, 
                                              config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, image_meta, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment, use_mini_mask=config.USE_MINI_MASK)
            
            # Skip images that have no instances. This can happen in cases 
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if np.sum(gt_boxes) <= 0:
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors, gt_boxes, config)

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(image.shape, random_rois, gt_boxes)
                if roi_predictions:
                    # Append two columns of zeros. TODO: needed?
                    rpn_rois = np.hstack([rpn_rois, np.zeros([rpn_rois.shape[0], 2], dtype=np.int32)])
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        build_image_proposal_targets(rpn_rois, gt_boxes, gt_masks, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros((batch_size,)+image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros((batch_size,)+image.shape, dtype=np.float32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 5), dtype=np.int32)
                if config.USE_MINI_MASK:
                    batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], 
                                               config.MAX_GT_INSTANCES))
                else:
                    batch_gt_masks = np.zeros((batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
                if random_rois:
                    batch_rpn_rois = np.zeros((batch_size,rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if roi_predictions:
                        batch_rois = np.zeros((batch_size,)+rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros((batch_size,)+mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros((batch_size,)+mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros((batch_size,)+mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:,:,ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = common.mold_image(image.astype(np.float32), config)
            batch_gt_boxes[b,:gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b,:,:,:gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois[:,:4]
                if roi_predictions:
                    batch_rois[b] = rois   # TODO: remove
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_boxes, batch_gt_masks]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])  # xxx breaks this notebook, but needed for the model
                    if roi_predictions:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
                        outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise





############################################################
#  Inferencing
############################################################


def process_image_detections(detections, mrcnn_mask, shape, window, config):
    """
    detections: [N, (y1, x1, y2, x2, class_id, score)]
    mrcnn_mask: [N, height, width, num_classes]
    
    shapes: the original shapes of the images (H, W, C)
    windows: the window in the image that includes the original 
             image (excludes padding). (y1, x1, y2, x2)

    Returns:
    boxes: [todo]
    class_ids:
    scores:
    masks: [height, width, num_instances]
    """
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:,4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
    
    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    # Compute scale and shift to translate coordinates to image domain.
    h_scale = shape[0] / (window[2] - window[0])
    w_scale = shape[1] / (window[3] - window[1])
    scale = min(h_scale, w_scale)  # TODO: is this right?
    shift = window[:2]  # y, x
    scales = np.array([scale, scale, scale, scale])
    shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
    
    # Translate bounding boxes to image domain
    boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
    
    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        mask = masks[i]
        # Resize mask to ROI size
        y1, x1, y2, x2 = boxes[i]
        mask = scipy.misc.imresize(mask, (y2-y1, x2-x1), interp='bilinear').astype(np.float32) / 255.0
        mask = np.where(mask >= 0.5, 1, 0).astype(np.uint8)
        # Put it in the right position in a bigger image-size mask.
        full_mask = np.zeros(shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
                 if full_masks else np.empty((0,) + masks.shape[1:3])
    
    return boxes, class_ids, scores, full_masks



class MaskRCNN():
    def __init__(self, mode, config):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and 
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']
        
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h/2**6 != int(h/2**6) or w/2**6 != int(w/2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        
        # Inputs
        input_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)
            # GT Boxes 
            # [batch, object count, (y1, x1, y2, x2, class_id)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 5], name="input_gt_boxes", dtype=tf.int32)
            # Normalize coordinates
            h, w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h, w, h, w, 1], axis=0), tf.float32)
            gt_boxes = KL.Lambda(lambda x: K.cast(x, tf.float32) / image_scale)(input_gt_boxes)
            # GT Masks
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None], 
                                        name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None], 
                                        name="input_gt_masks", dtype=bool)
                
        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = resnet_graph(input_image, "resnet101", stage5=True)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in the config.
        P5 = KL.Conv2D(256, (1, 1), **layer_params(), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(256, (1, 1), **layer_params(), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(256, (1, 1), **layer_params(), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), **layer_params(), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(256, (3, 3), padding="SAME", **layer_params(), name="fpn_p2")(P2)
        P3 = KL.Conv2D(256, (3, 3), padding="SAME", **layer_params(), name="fpn_p3")(P3)
        P4 = KL.Conv2D(256, (3, 3), padding="SAME", **layer_params(), name="fpn_p4")(P4)
        P5 = KL.Conv2D(256, (3, 3), padding="SAME", **layer_params(), name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
        
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        
        # Generate Anchors
        self.anchors = common.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, 
                                                       config.RPN_ANCHOR_RATIOS,
                                                       config.BACKBONE_SHAPES,
                                                       config.BACKBONE_STRIDES, 
                                                       config.RPN_ANCHOR_STRIDE)    
        
        # RPN
        # rpn_score = [batch, height, width, num_bbox, 2]
        # rpn_bbox = [batch, height, width, num_bbox, 5]  # TODO: why 5?
        # rpn_rois = [batch, num_rois, (y1, x1, y2, x2)] in normalized coordinates

        # RPN Model
        rpn = build_rpn_model(config, len(config.RPN_ANCHOR_RATIOS), 256)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists 
        # of outputs across levels. 
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) 
                    for o, n in zip(outputs, output_names)]
        
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Select proposals
        # TODO: rename ProposalLayer to ProposalSubsetLayer
        # Proposals are [N, (y1, x1, y2, x2)] in normalized coordinates.
        post_nms_top_n = config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=post_nms_top_n, nms_threshold=0.7, 
                                name="ROI", anchors=self.anchors, config=config)([rpn_class, rpn_bbox])
        
        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            _, _, _, class_id_mask = KL.Lambda(lambda x: common.parse_image_meta_graph(x), 
                                            mask=[None, None, None, None])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                    name="input_roi", dtype=np.int32)
                # Normalize coordinates to 0-1 range.
                target_rois = KL.Lambda(lambda x: K.cast(x, tf.float32) / image_scale[:4])(input_rois)
            else:
                target_rois = rpn_rois

            # Subsample RPN ROIs before passing them to classification and mask heads.
            # Note that ROIs might be padded with zeros if there aren't enough. todo: are they?
            rois, target_class_ids, target_bbox, target_mask =\
                ProposalTargetLayer(config, name="proposal_targets")([target_rois, gt_boxes, input_gt_masks])

            # Network Heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                build_fpn_classifier_graph(rois, mrcnn_feature_maps, config)
                
            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps, config)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, class_id_mask])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta, 
                    input_rpn_match, input_rpn_bbox, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                    mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                    rpn_rois, output_rois,
                    rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                build_fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, config)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = RefinementLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            # todo: keep coordinates in normalized form to avoid repeated conversions
            h, w = config.IMAGE_SHAPE[:2]
            detection_boxes = KL.Lambda(lambda x: x[...,:4]/np.array([h, w, h, w]))(detections)
            
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps, config)

            model = KM.Model([input_image, input_image_meta], 
                        [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox], 
                        name='mask_rcnn')
            
        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            model = multigpunb.ParallelModel(model, config.GPU_COUNT)
        
        return model

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of the ability to exclude some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        layers = self.keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)
        
        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        # Pre-defined layer regular expressions
        layer_regex = {
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",  # all but the backbone
            "all": "todo",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True, 
                                                batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True, 
                                            batch_size=self.config.BATCH_SIZE)

        # Start a new model (unless overwritten if starting from a TRAINED_MODEL)
        self.epoch = 0
        now = datetime.datetime.now()

        # Directory for training logs
        log_dir = os.path.join(MODEL_DIR, "{}{:%Y%m%d%H}".format(self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        checkpoint_path = os.path.join(log_dir, "mask_rcnn_*epoch*.h5")
        checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")
        print("Checkpoint Path: ", checkpoint_path)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_dir, 
                                        histogram_freq=0, write_graph=True, write_images=False),
            # TODO: add validation loss monitoring and saving best only
            keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                            verbose=0, save_weights_only=True),
            # todo: APHistory(self.config),
        ]
        
        # Common parameters to pass to fit_generator()
        fit_kwargs = {
            "steps_per_epoch": self.config.STEPS_PER_EPOCH,
            "callbacks": callbacks,
            "validation_data": next(val_generator),
            "validation_steps": self.config.VALIDATION_STPES,
            "max_queue_size": 32,
            "workers": max(self.config.BATCH_SIZE // 2, 2),
            "use_multiprocessing": True,
        }
        
        # Train
        print("Starting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        set_trainable(self.keras_model, layers)
        compile_model(self.keras_model, learning_rate, self.config.LEARNING_MOMENTUM, self.config)

        # todo: Experimentally, loss reductions slows down ater 40K images (50 epoch * 100 stpes/epoch * 8 images/batch).
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            **fit_kwargs
            )
        self.epoch = max(self.epoch, epochs)
            
        
    # todo: keep this or the one in common.py?
    # def mold_image(self, images):
    #     """Takes RGB images with 0-255 values and subtraces
    #     the mean pixel and converts it to float. Expects image
    #     colors in RGB order.
    #     """
    #     return images.astype(np.float32) - self.config.MEAN_PIXEL

    # def unmold_image(self, molded_images):
    #     """Takes a image normalized with mold() and returns the original."""
    #     return (molded_images + self.config.MEAN_PIXEL).astype(np.uint8)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.
        
        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # todo: should resizing go inside mold_image()?
            molded_image, window, scale, padding = common.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = common.mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = common.compose_image_meta(
                0, image.shape, window, 
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def detect(self, images, verbose=0):
        """
        List of images, potentially of different sizes.
        """
        if verbose: 
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
        # Run object detection
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
        rois, rpn_class, rpn_bbox =\
            self.keras_model.predict([molded_images, image_metas], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                process_image_detections(detections[i], mrcnn_mask[i], 
                                         image.shape, windows[i], self.config)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results


    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already 
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))
        
        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers
    
    def run_graph(self, images, outputs):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        outputs: List of tuples (name, tensor) to compute. The tensors are 
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Run inference
        molded_images, image_metas, windows = self.mold_inputs(images)
        # todo: support training mode?
        # if TEST_MODE == "training":
        #     model_in = [molded_images, image_metas, 
        #                 target_rpn_match, target_rpn_bbox, 
        #                 gt_boxes, gt_masks]
        #     if not config.USE_RPN_ROIS:
        #         model_in.append(target_rois)
        #     if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        #         model_in.append(1.)
        #     outputs_np = kf(model_in)
        # else:

        model_in = [molded_images, image_metas]
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v) for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np



############################################################
#  Keras Overrides
############################################################


def load_weights(self, filepath, by_name=False, exclude=None):
    """Modified version of the correspoding Keras function with
    the addition of the ability to exclude some layers from loading.
    exlude: list of layer names to excluce
    """
    import h5py
    from keras.engine import topology

    if h5py is None:
        raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    layers = self.layers

    # Exclude some layers
    if exclude:
        layers = filter(lambda l: l.name not in exclude, layers)
    
    if by_name:
        topology.load_weights_from_hdf5_group_by_name(f, layers)
    else:
        topology.load_weights_from_hdf5_group(f, layers)
    if hasattr(f, 'close'):
        f.close()

