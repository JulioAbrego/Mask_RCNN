"""
Mask R-CNN
Unit tests for model.py

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
import numpy as np
import tensorflow as tf
import scipy.misc
import scipy.io
import skimage.io

import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
from keras.engine import Layer
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import Layer

import common
import model as modellib

# Proposal Layer

# Unit Test
# todo: remove if possible
class TestConfig(common.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    BATCH_SIZE = 2
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 2
    ARCHITECTURE = "resnet101_fpn"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
config = TestConfig()

anchors = common.generate_pyramid_anchors(
    scales=(32, 64, 128, 256, 512),
    ratios=[0.5, 1, 2],
    feature_shapes=config.BACKBONE_SHAPES,
    feature_strides=[4, 8, 16, 32, 64],
    anchor_stride=2,
    )
num_anchors = anchors.shape[0]
rpn_class = KL.Input(shape=[num_anchors, 2])
rpn_bbox = KL.Input(shape=[num_anchors, 4])
proposals = modellib.ProposalLayer(1, 0.7, anchors, config)([rpn_class, rpn_bbox])
kf = K.function([rpn_class, rpn_bbox], [proposals])
# Run Tests
rpn_class_np = np.zeros([2, num_anchors, 2])
rpn_class_np[:, 4:8, 1] = np.array([
    [0, .3, 0, .4],
    [.6, 0, .5, 0],
])
rpn_bbox_np = np.random.uniform(-3, 3, size=[2, num_anchors, 4])
rpn_bbox_np[:, 4:8] = np.array([
    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
    [[.1, .1, .1, .1], [.2, .2, .2, .2], [.3, .3, .3, .3], [.4, .4, .4, .4]],
])
result = kf([rpn_class_np, rpn_bbox_np])[0]
assert np.allclose(result, np.array([
    [[ 0., 0., 0.04727408, 0.06289908]],
    [[ 0., 0., 0.01625315, 0.02406565]]]))


# ROIAlign

# Unit Test
boxes = KL.Input(shape=[None, 4])
p2 = KL.Input(shape=[8, 8, 1], name="p2")
p3 = KL.Input(shape=[4, 4, 1], name="p3")
p4 = KL.Input(shape=[2, 2, 1], name="p4")
p5 = KL.Input(shape=[1, 1, 1], name="p5")
kf = K.function([boxes, p2, p3, p4, p5], 
                [modellib.PyramidROIAlign((2, 2), config.IMAGE_SHAPE)([boxes, p2, p3, p4, p5])])

# Run Tests
ih, iw = config.IMAGE_SHAPE[:2]
boxes_np = np.array([[
    [0, 0, 20/ih, 20/iw],   # level 2
    [0, 0, 40/ih, 20/iw],   # level 2
    [0, 0, 100/ih, 100/iw], # level 3
    [0, 0, 224/ih, 224/iw], # level 4
    [0, 0, 448/ih, 448/iw], # level 5
    [0, 0, ih/ih, iw/iw],   # level 6, max size
    [0, 0, 112/ih, 112/iw], # level 3
]])
p2_np = np.ones([1, 8, 8, 1]) * 2
p3_np = np.ones([1, 4, 4, 1]) * 3
p4_np = np.ones([1, 2, 2, 1]) * 4
p5_np = np.ones([1, 1, 1, 1]) * 5
result = kf([boxes_np, p2_np, p3_np, p4_np, p5_np])[0]

assert result.shape == (1, 7, 2, 2, 1)
assert np.allclose(result[0, 0, :, :, 0], np.array([[2, 2], [2, 2]]))
assert np.allclose(result[0, 1, :, :, 0], np.array([[2, 2], [2, 2]]))
assert np.allclose(result[0, 2, :, :, 0], np.array([[3, 3], [3, 3]]))
assert np.allclose(result[0, 3, :, :, 0], np.array([[4, 4], [4, 4]]))
assert np.allclose(result[0, 4, :, :, 0], np.array([[5, 5], [5, 5]]))
assert np.allclose(result[0, 5, :, :, 0], np.array([[5, 5], [5, 5]]))
assert np.allclose(result[0, 6, :, :, 0], np.array([[3, 3], [3, 3]]))


# Loss Functions

# Smooth L1 Loss
y_true = KL.Input(shape=[None])
y_pred = KL.Input(shape=[None])
kf = K.function([y_true, y_pred], [modellib.smooth_l1_loss(y_true, y_pred)])
# Run Tests
y = np.array([[0, 1, 2, 1, .2]])
p = np.array([[0, 0, 2, 2, .1]])
assert np.allclose(kf([y, p])[0], np.array([[0, .5, 0, .5, .005]]), 1e-5)

# rpn_class_loss
rpn_match = KL.Input(shape=[None, 1])
rpn_class_logits = KL.Input(shape=[None, 2])
kf = K.function([rpn_match, rpn_class_logits], [modellib.rpn_class_loss_graph(rpn_match, rpn_class_logits)])
# Run Tests
assert kf([np.array([[[0], [0]]]), np.array([[[0, 1], [1, 0]]])])[0] == 0
assert kf([np.array([[[1], [1]]]), np.array([[[0, 20], [0, 25]]])])[0] == 0
assert kf([np.array([[[-1], [-1]]]), np.array([[[20, -10], [10, -10]]])])[0] == 0
assert np.allclose(kf([np.array([[[1], [-1]]]), np.array([[[0, 1], [0, 1]]])])[0], 0.81326163)
assert np.allclose(
    kf([np.array([ [[1], [-1]], [[1], [-1]] ]), 
    np.array([ [[0, 5], [5, 0]], [[0, 5], [5, 0]] ])])[0], 0.00671535)


# rpn_bbox_loss
# Unit Test
class TestConfig(common.CocoConfig):
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1
test_config = TestConfig()

target_bbox = KL.Input(shape=[None, 4])
rpn_match = KL.Input(shape=[None, 1])
rpn_bbox = KL.Input(shape=[None, 4])
kf = K.function([target_bbox, rpn_match, rpn_bbox], 
                [modellib.rpn_bbox_loss_graph(test_config, target_bbox, rpn_match, rpn_bbox)])
# Run tests
assert kf([np.array([[[0, 0, 0, 0]]]), np.array([[[0]]]), np.array([[[0, 0, 0, 0]]])])[0] == 0
assert kf([np.array([[[0, 0, 0, 0]]]), np.array([[[-1]]]), np.array([[[0, 0, 0, 0]]])])[0] == 0
assert kf([np.array([[[1, 1, 1, 1]]]), np.array([[[1]]]), np.array([[[1, 1, 1, 1]]])])[0] == 0
assert kf([np.array([[[0, 0, 0, 0]]]), np.array([[[1]]]), np.array([[[-1, -1, 1, 1]]])])[0] == 0.5
assert kf([np.array([[[0, 0, 0, 0]]]), np.array([[[1]]]), np.array([[[2, 2, -2, -2]]])])[0] == 1.5
assert kf([np.array([[[0, 0, 0, 0]]]), np.array([[[1]]]), np.array([[[.5, .5, .5, .5]]])])[0] == 0.125
# todo: assert kf([np.array([[[0, 0, 0, 0], [0, 0, 0, 0]]]), 
#     np.array([[[1], [0], [1]]]), 
#     np.array([[[0.2, 0.3, 0, 0], [0, 0, 0, 0], [0, 0, 0, .9]]]) ])[0] == 0.05875
# todo: assert kf([np.array([[[0, 0, 0, 0]], [[0, 0, 0, 0]]]), 
#     np.array([[[1], [0]], [[0], [1]]]), 
#     np.array([[[0.2, 0.3, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, .9]]]) ])[0] == 0.05875


# mrcnn_class_loss_graph
# Unit Test
target_class_ids = KL.Input(shape=[2])
pred_class_logits = KL.Input(shape=[2, 3])
class_id_mask = KL.Input(shape=[3])
loss = modellib.mrcnn_class_loss_graph(target_class_ids, pred_class_logits, class_id_mask)
kf = K.function([target_class_ids, pred_class_logits, class_id_mask], [loss])
# Run
target_np = np.array([ [0, 1], [2, 1] ], dtype=np.int64)
output_np = np.array([[[1, 0, 0], 
                       [0, 1, 0]], 
                      [[0, 0, 1], 
                       [0, 1, 0]]], dtype=np.float32)
# todo: assert kf([target_np, output_np, np.array([[1, 1, 1]])])[0] == 0.55144465
# todo: assert kf([target_np, output_np, np.array([[1, 0, 1]])])[0] == 0.55144465


# mrcnn_bbox_loss
# Unit Test
target_bbox = KL.Input(shape=[2, 4])
target_class_ids = KL.Input(shape=[2, 1])
pred_bbox = KL.Input(shape=[2, 3, 4])
kf = K.function([target_bbox, target_class_ids, pred_bbox], 
                [modellib.mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox)])
# Run Tests
class_ids = np.array([[[2], [0]], [[1], [0]]])
y = np.array([
    [[.2, 0, 0, 0],
     [0, .1, 0, 0]],
    [[0, .1, 0, 0],
     [0, 0, 0, 0]],
])
p = np.arange(48).reshape((2, 2, 3, 4))
assert np.allclose(kf([y, class_ids, p])[0], 18.96250153)


# mrcnn_mask_loss
# Unit Test
target_masks = KL.Input(shape=[None, 3, 3])
target_class_ids = KL.Input(shape=[None])
pred_masks = KL.Input(shape=[None, 3, 3, None])
kf = K.function([target_masks, target_class_ids, pred_masks], 
                [modellib.mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks)])
# Run Tests
assert kf([np.zeros((1, 1, 3, 3)), np.array([[1]]), np.zeros((1, 1, 3, 3, 2))])[0] < 1e-5
assert kf([np.ones((1, 2, 3, 3)), np.array([[1, 0]]), np.ones((1, 2, 3, 3, 2))])[0] < 1e-5
assert kf([np.ones((2, 2, 3, 3)), np.array([[1, 0], [0, 1]]), np.ones((2, 2, 3, 3, 2))])[0] < 1e-5

assert np.allclose(kf([np.ones((2, 1, 3, 3)), np.array([[1], [1]]), np.zeros((2, 1, 3, 3, 2))])[0], 16.1180954)
assert np.allclose(kf([np.ones((2, 1, 3, 3)), np.array([[0], [1]]), np.zeros((2, 1, 3, 3, 2))])[0], 16.1180954)
assert np.allclose(kf([np.ones((2, 1, 3, 3)), np.array([[1], [2]]), np.zeros((2, 1, 3, 3, 4))])[0], 16.1180954)
assert np.allclose(kf([np.ones((2, 2, 3, 3)), np.array([[1, 0], [0, 2]]), 
                       np.zeros((2, 2, 3, 3, 4))])[0], 16.1180954)
y = np.ones((1, 1, 3, 3))
c = np.array([[1]])
p = np.zeros((1, 1, 3, 3, 2))
kf([y, c, p])[0]
assert abs(kf([y, c, p])[0] - 16.1180954) < 1e-5

y = np.zeros((1, 1, 3, 3))
c = np.array([[1]])
p = np.zeros((1, 1, 3, 3, 2))
p[0, 0, :, :, 1] = 1
assert abs(kf([y, c, p])[0] - 15.94238472) < 1e-5

y_true = KL.Input(shape=[None, 3, 3])
y_true_class_ids = KL.Input(shape=[None])
y_pred = KL.Input(shape=[None, 3, 3, 4])
kf = K.function([y_true, y_true_class_ids, y_pred], [modellib.mrcnn_mask_loss_graph(y_true, y_true_class_ids, y_pred)])


# batch_pack  todo: right place?
# Unit test
with tf.Graph().as_default():
    with tf.Session() as session:
        x_ph = tf.placeholder(tf.int32, (None, 4))
        count_ph = tf.placeholder(tf.int32, (None,))
        op = modellib.batch_pack(x_ph, count_ph, 5)
        
        result = session.run(op, { 
            x_ph: np.arange(20).reshape([5, 4]),
            count_ph: np.array([2, 0, 3, 0, 1])
        })
        assert np.allclose(result, np.array([ 0, 1, 8, 9, 10, 16]))


# Proposal Target Layer

# Unit Tests - Full Masks
config.IMAGE_SHAPE = [100, 100, 3]
config.MASK_SHAPE = [10, 10]
config.USE_MINI_MASK = False

num_rois = 5
num_gt = 3
rpn_rois = KL.Input([num_rois, 4], dtype=tf.float32)
gt_boxes = KL.Input([num_gt, 5], dtype=tf.float32)
gt_masks = KL.Input([None, None, num_gt], dtype=tf.bool)  # height, width, instances
x = modellib.ProposalTargetLayer(config)([rpn_rois, gt_boxes, gt_masks])
kf = K.Function([rpn_rois, gt_boxes, gt_masks], x)

# Run tests
rpn_rois_np = np.array([[
    [ 0,  0, .65, .65], 
    [.1, .1, .5,  .5],
    [.7, .7,  1,  1],
    [ 0,  0,  1,  1],
    [ 0,  0,  0,  0],  # empty ROI
    ]])
gt_boxes_np = np.array([[
    [ 0,   0, .5, .5, 1],
    [.75, .75, 1,  1, 2],
    [ 0,   0,  0,  0, 0],  # empty GT box
    ]])
gt_masks_np = np.zeros([1, 100, 100, num_gt], dtype=bool)
gt_masks_np[0, 0:50,  10:40,  0] = 1
gt_masks_np[0, 80:95, 75:100, 1] = 1

result = kf([rpn_rois_np, gt_boxes_np, gt_masks_np])

# ROIs. Order is random. Sort them for correct comparison.
ix = np.argsort(np.sum(result[0][0], axis=1))
assert np.allclose(result[0][0, ix], np.array([
    [ 0.1,  0.1, 0.5, 0.5],
    [ 0.,  0.,  0.64999998,  0.64999998],
    [ 0.,  0.,  1.,  1.],
    [ 0.69999999, 0.69999999,  1.,  1.],
]))

# class IDs
assert np.allclose(result[1][0, ix], np.array([1, 1, 0, 2]))

# BBox deltas
assert np.allclose(result[2][0, ix], np.array([
    [-1.25000024, -1.25000024,  1.11571777,  1.11571777,  1.],
    [-1.15384603, -1.15384603, -1.31182122, -1.31182122,  1.],
    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.],
    [ 0.83333254,  0.83333254, -0.91160786, -0.91160786,  2.],
]))


# Unit Tests - Mini Masks

num_rois = 5
num_gt = 3
rpn_rois = KL.Input([num_rois, 4], dtype=tf.float32)
gt_boxes = KL.Input([num_gt, 5], dtype=tf.float32)
gt_masks = KL.Input([None, None, num_gt], dtype=tf.bool)  # height, width, instances
x = ProposalTargetLayer(config)([rpn_rois, gt_boxes, gt_masks])
kf = K.Function([rpn_rois, gt_boxes, gt_masks], x)

# Run tests
config.IMAGE_SHAPE = [100, 100, 3]
config.MASK_SHAPE = [10, 10]
config.USE_MINI_MASK = True
config.MINI_MASK_SHAPE = (20, 20)

rpn_rois_np = np.array([[
    [ 0,  0, .65, .65], 
    [.1, .1, .5,  .5],
    [.7, .7,  1,  1],
    [ 0,  0,  1,  1],
    [ 0,  0,  0,  0],  # empty ROI
    ]])
gt_boxes_np = np.array([[
    [ 0,   0, .5, .5, 1],
    [.75, .75, 1,  1, 2],
    [ 0,   0,  0,  0, 0],  # empty GT box
    ]])
gt_masks_np = np.zeros([1, 20, 20, num_gt], dtype=bool)
gt_masks_np[0, 0:20, 4:16, 0] = 1
gt_masks_np[0, 4:16, 0:20, 1] = 1

result = kf([rpn_rois_np, gt_boxes_np, gt_masks_np])

# ROIs. Order is random. Sort them for correct comparison.
ix = np.argsort(np.sum(result[0][0], axis=1))
assert np.allclose(result[0][0, ix], np.array([
    [ 0.1,  0.1, 0.5, 0.5],
    [ 0.,  0.,  0.64999998,  0.64999998],
    [ 0.,  0.,  1.,  1.],
    [ 0.69999999, 0.69999999,  1.,  1.],
]))

# class IDs
assert np.allclose(result[1][0, ix], np.array([1, 1, 0, 2]))

# BBox deltas
assert np.allclose(result[2][0, ix], np.array([
    [-1.25000024, -1.25000024,  1.11571777,  1.11571777,  1.],
    [-1.15384603, -1.15384603, -1.31182122, -1.31182122,  1.],
    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.],
    [ 0.83333254,  0.83333254, -0.91160786, -0.91160786,  2.],
]))
