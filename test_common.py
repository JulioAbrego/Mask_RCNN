"""
Mask R-CNN
Unit tests for common.py

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""


# Unit Test
with tf.Graph().as_default():
    with tf.Session() as session:
        box_ph = tf.placeholder(tf.int32, shape=[None, 4])
        gt_box_ph = tf.placeholder(tf.int32, shape=[None, 4])
        result_ph = box_refinement_graph(box_ph, gt_box_ph)

        # Test values
        box = np.array([[20, 20,  60, 60],
                        [10, 10,  50, 50],
                        [70, 70, 100, 100],
                        [0,   0, 100, 100],
                        ])
        gt_box = np.array([[20, 20,  60,  60],
                           [0,   0,  60,  60],
                           [75, 75, 100, 100],
                           [75, 75, 100, 100],
                           ])
        result = session.run(result_ph, {
            box_ph: box,
            gt_box_ph: gt_box,
        })

        assert np.allclose(result, np.array([
            [0.,  0.,  0.,  0.],
            [0.,  0.,  0.4054651,  0.4054651],
            [0.08333334,  0.08333334, -0.18232158, -0.18232158],
            [0.375,  0.375, -1.38629436, -1.38629436]
        ]), 1e-5)


# Unit Test
compute_box_refinement(
    np.array([[0, 0, 100, 100]]), np.array([[0, -10, 200, 90]]))


with tf.Graph().as_default():
    with tf.Session() as session:
        input_ph = tf.placeholder(tf.int32, (None, 4))
#         op = tf.nn.top_k(input_ph, 2, sorted=True).indices
#         op = tf.gather_nd(input_ph, op)

        def top_k(x):
            op = tf.nn.top_k(x, 2, sorted=True).indices
            op = tf.Print(op, [op], summarize=300)
            op = tf.gather(x, op)
            return op
        op = batch_slice(input_ph, top_k, 2)

#         op = tf.gather(input_ph, [0, 2])
#         op = batch_slice(input_ph, lambda x: tf.gather(x[0], [0, 2]), 2)

        result = session.run(op, {
            input_ph: [[2, 0, 1, 3], [10, 15, 20, 30]]
        })
        print(result)


# Unit Test
# Simplest
a = generate_anchors([1], [1], (1, 1), 1, 1)
assert np.array_equal(a, [[-0.5, -0.5, 0.5, 0.5]])
# 1 scale, 2 ratios
a = generate_anchors([32], [1, 4], (1, 1), 16, 1)
assert np.array_equal(a, [[-16, -16,  16,  16], [-8, -32, 8,  32]])
# 2 scales, 2 ratios
a = generate_anchors([32, 64], [1, .25], (1, 1), 16, 1)
assert np.array_equal(a, [[-16, -16,  16,  16, ],
                          [-32, -32,  32,  32, ],
                          [-32,  -8,  32,   8, ],
                          [-64, -16,  64,  16, ]])
# 1 scales, 1 ratios, 2x2 feature map
a = generate_anchors([32], [1], (2, 2), 16, 1)
assert np.array_equal(a, [[-16, -16,  16, 16],
                          [-16,   0,  16, 32],
                          [0, -16,  32, 16],
                          [0,   0,  32, 32]])
# Pyramid with 2 scales, 2 ratios
a = generate_pyramid_anchors([32, 64], [1, .25], [(2, 2), (1, 1)], [16, 32], 1)
assert np.array_equal(a, [[-16, -16,  16,  16, ],
                          [-32,  -8,  32,   8, ],
                          [-16,   0,  16,  32, ],
                          [-32,   8,  32,  24, ],
                          [0, -16,  32,  16, ],
                          [-16,  -8,  48,   8, ],
                          [0,   0,  32,  32, ],
                          [-16,   8,  48,  24, ],
                          [-32, -32,  32,  32, ],
                          [-64, -16,  64,  16, ]])

# Pyramid with 2 scales, 2 ratios, and anchor stride of 2.
a = generate_pyramid_anchors([32, 64], [1, .25], [(2, 2), (1, 1)], [16, 32], 2)
assert np.array_equal(a, [[-16, -16,  16,  16, ],
                          [-32,  -8,  32,   8, ],
                          [-32, -32,  32,  32, ],
                          [-64, -16,  64,  16, ]])
