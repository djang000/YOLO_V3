import os, sys, datetime, re
import numpy as np
import scipy.misc
from collections import namedtuple
# import gen_cocodb as datapipe
import tensorflow as tf
import tensorflow.contrib.layers as layer
import tensorflow.contrib.slim as slim

############################################################
#  visualize Functions
############################################################
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


############################################################
#  YOLOV3 Class
############################################################

def BatchNorm(inputs, epsilon=1e-3, suffix=''):
    """
       Assuming TxHxWxC dimensions on the tensor, will normalize over
       the H,W dimensions. Use this before the activation layer.
       This function borrows from:
           http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

       Note this is similar to batch_normalization, which normalizes each
       neuron by looking at its statistics over the batch.

       :param input_:
           input tensor of NHWC format
       """
    # Create scale + sx               hift. Exclude batch dimension.
    stat_shape = inputs.get_shape().as_list()
    scale = tf.get_variable('scale' + suffix,
                            initializer=tf.ones(stat_shape[3]))
    shift = tf.get_variable('shift' + suffix,
                            initializer=tf.zeros(stat_shape[3]))

    means, vars = tf.nn.moments(inputs, axes=[1, 2],
                                          keep_dims=True)
    # Normalization
    inputs_normed = (inputs - means) / tf.sqrt(vars + epsilon)

    # Perform trainable shift.
    output = tf.add(tf.multiply(scale, inputs_normed), shift, name=suffix)
    print(output)

    return output

def Conv2D(data, out_ch, kernel, stride, padding="SAME", name=None, activation=None, is_training=True ):
    in_ch = data.get_shape().as_list()[-1]
    W = tf.get_variable(name="{}_W".format(name),
                        shape=[kernel, kernel, in_ch, out_ch], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1e-3),
                        trainable=is_training)
    feature = tf.nn.conv2d(data, W, strides=[1, stride, stride, 1], padding=padding, name=name)
    if activation is "relu":
        feature =tf.nn.relu(feature, name="{}_relu".format(name))
    return feature

def training_scope(is_training=True,
                   weight_decay=0.00004,
                   stddev=0.09,
                   dropout_keep_prob=0.8):
    """Defines Mobilenet training scope.
    Usage:
        with tf.contrib.slim.arg_scope(mobilenet.training_scope()):
        logits, endpoints = mobilenet_v2.mobilenet(input_tensor)
        # the network created will be trainble with dropout/batch norm
        # initialized appropriately.
    Args:
        is_training: if set to False this will ensure that all customizations are
            set to non-training mode. This might be helpful for code that is reused
            across both training/evaluation, but most of the time training_scope with
            value False is not needed. If this is set to None, the parameters is not
            added to the batch_norm arg_scope.
        weight_decay: The weight decay to use for regularizing the model.
        stddev: Standard deviation for initialization, if negative uses xavier.
        dropout_keep_prob: dropout keep probability (not set if equals to None).
        bn_decay: decay for the batch norm moving averages (not set if equals to None).
    Returns:
        An argument scope to use via arg_scope.
    """
    # Note: do not introduce parameters that would change the inference
    # model here (for example whether to use bias), modify conv_def instead.
    batch_norm_params = {
        'decay': 0.997,
        'is_training': is_training
    }
    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
        weights_initializer=weight_intitializer,
        biases_initializer=None,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
        padding='SAME') :
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as sc:
                    return sc

Conv = namedtuple('Conv', ['kernel', 'stride', 'channel'])
InvertedBottleneck = namedtuple('InvertedBottleneck', ['up_sample', 'channel', 'stride'])

# Sequence of layers, described in Table 2
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, channel=32),              # 1 block, input 416x416x3
    InvertedBottleneck(up_sample=1, channel=16, stride=1),  # 2 block, input : 208x208x32

    InvertedBottleneck(up_sample=6, channel=24, stride=2),  # 3 block, input: 208x208x16
    InvertedBottleneck(up_sample=6, channel=24, stride=1),  # 4 block, input: 104x104x24

    InvertedBottleneck(up_sample=6, channel=32, stride=2),  # 5 block, input: 104x104x24
    InvertedBottleneck(up_sample=6, channel=32, stride=1),  # 6 block, input: 52x52x32
    InvertedBottleneck(up_sample=6, channel=32, stride=1),  # 7 block, input: 52x52x32

    InvertedBottleneck(up_sample=6, channel=64, stride=2),  # 8 block, input: 52x52x32
    InvertedBottleneck(up_sample=6, channel=64, stride=1),  # 9 block, input: 26x26x64
    InvertedBottleneck(up_sample=6, channel=64, stride=1),  # 10 block, input: 26x26x64
    InvertedBottleneck(up_sample=6, channel=64, stride=1),  # 11 block, input: 26x26x64

    InvertedBottleneck(up_sample=6, channel=96, stride=1),  # 12 block, input: 26x26x64
    InvertedBottleneck(up_sample=6, channel=96, stride=1),  # 13 block, input: 26x26x96
    InvertedBottleneck(up_sample=6, channel=96, stride=1),  # 14 block, input: 26x26x96

    InvertedBottleneck(up_sample=6, channel=160, stride=2),  # 15 block, input: 26x26x96
    InvertedBottleneck(up_sample=6, channel=160, stride=1),  # 16 block, input: 13x13x160
    InvertedBottleneck(up_sample=6, channel=160, stride=1),  # 17 block, input: 13x13x160

    InvertedBottleneck(up_sample=6, channel=320, stride=1),  # 18 block, input: 13x13x160

    Conv(kernel=[1, 1], stride=1, channel=1280),             # 19 block, input: 13x13x320
]

def backbone_graph(inputs):
    depth = lambda d: max(int(d * 1.0), 8)

    end_points = {}
    net_lists=[]

    with tf.variable_scope('MobilenetV2', [inputs], reuse=False) as sc:
        net = inputs
        for i, conv_def in enumerate(_CONV_DEFS):
            stride = conv_def.stride
            if isinstance(conv_def, Conv):
                name = 'Conv2d_%d' % i
                num_channel = depth(conv_def.channel)
                net = slim.conv2d(net, num_channel,
                                    conv_def.kernel,
                                  activation_fn=tf.nn.relu6,
                                  stride=stride)
                end_points[name] = net
                print(net)
                net_lists.append(net)
            elif isinstance(conv_def, InvertedBottleneck):
                if i == 1:
                    scope = 'expanded_conv'
                else:
                    scope = 'expanded_conv_%d' % (i - 1)
                input_tensor = net

                with tf.variable_scope(scope):
                    net_ch = input_tensor.get_shape().as_list()[-1]
                    inner_size = conv_def.up_sample * net_ch
                    if inner_size > net_ch:
                        net = slim.conv2d(net, inner_size, [1, 1], stride=1, activation_fn=tf.nn.relu6,
                                          scope='expand')
                        end_points[scope + '/expand'] = net
                        print(net)

                        net_lists.append(net)

                    net = slim.separable_conv2d(net, num_outputs=None,
                                                kernel_size=[3, 3],
                                                depth_multiplier=1,
                                                stride=stride,
                                                activation_fn=tf.nn.relu6,
                                                scope='depthwise')
                    end_points[scope + '/depthwise'] = net
                    print(net)
                    net_lists.append(net)

                    net = slim.conv2d(net, conv_def.channel, [1, 1], stride=1, activation_fn=tf.identity,
                                      scope='project')
                    end_points[scope + '/project'] = net
                    print(net)

                    net_lists.append(net)

                    if stride == 1 and net_ch == net.get_shape().as_list()[-1]:
                        net += input_tensor
                        net = tf.identity(net, name='output')

        # for i, k in end_points.items():
        #     print(i, k)

    return end_points


def get_output_layer(x, num_filters, out_filters, scope):
    with tf.variable_scope(scope):
        x = slim.conv2d(x, num_filters, [1, 1], stride=1, activation_fn=tf.nn.relu6, scope='conv_1')
        x = slim.conv2d(x, num_filters*2, [3, 3], stride=1, activation_fn=tf.nn.relu6, scope='conv_2')
        x = slim.conv2d(x, num_filters, [1, 1], stride=1, activation_fn=tf.nn.relu6, scope='conv_3')
        x = slim.conv2d(x, num_filters*2, [3, 3], stride=1, activation_fn=tf.nn.relu6, scope='conv_4')
        y1 = slim.conv2d(x, num_filters, [1, 1], stride=1, activation_fn=tf.nn.relu6, scope='conv_5')

        x = slim.conv2d(y1, num_filters*2, [3, 3], stride=1, activation_fn=tf.nn.relu6, scope='conv_6')
        y2 = slim.conv2d(x, out_filters, [1, 1], stride=1, normalizer_fn=None,
                         activation_fn=None, scope='conv_7')
    return y1, y2

def concat(x, y, num_filters, scope ):
    with tf.variable_scope(scope):
        h, w = y.get_shape().as_list()[1:3]
        net = slim.conv2d(x, num_filters, [1, 1], stride=1, activation_fn=tf.nn.relu6, scope='conv_1x1' )
        net = tf.image.resize_images(net, [h, w])
        out = tf.concat([net, y], axis=-1)
    return out


def head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    feat_dtype = feats.dtype
    anchors_tensors = tf.cast(tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2]), dtype=feat_dtype)

    grid_shape = tf.shape(feats)[1:3]
    grid_y = tf.tile(tf.reshape(tf.range(0, limit=grid_shape[0], delta=1), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])

    grid_x = tf.tile(tf.reshape(tf.range(0, limit=grid_shape[1], delta=1), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid = tf.cast(tf.concat([grid_x, grid_y], axis=-1), dtype=feat_dtype)

    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.nn.sigmoid(feats[..., :2])+ grid) / tf.cast(grid_shape[::-1], dtype=feat_dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensors / tf.cast(input_shape[::-1], dtype=feat_dtype)
    box_confidence = tf.nn.sigmoid(feats[..., 4:5])
    box_class_probs = tf.nn.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh

    return box_xy, box_wh, box_confidence, box_class_probs


def switch(condition, then_expression, else_expression):
    if condition.dtype != tf.bool:
        condition = tf.cast(condition, 'bool')

    def ndim(x):
        dims = x.get_shape()._dims
        if dims is not None:
            return len(dims)
        return None

    cond_ndim = ndim(condition)
    if not cond_ndim:
        if not callable(then_expression):
            def then_expression_fn():
                return then_expression
        else:
            then_expression_fn = then_expression
        if not callable(else_expression):
            def else_expression_fn():
                return else_expression
        else:
            else_expression_fn = else_expression
        x = tf.cond(condition,
                    then_expression_fn,
                    else_expression_fn)
    else:
        # tf.where needs its condition tensor
        # to be the same shape as its two
        # result tensors
        if callable(then_expression):
            then_expression = then_expression()
        if callable(else_expression):
            else_expression = else_expression()
        expr_ndim = ndim(then_expression)
        if cond_ndim > expr_ndim:
            raise ValueError('Rank of `condition` should be less than or'
                             ' equal to rank of `then_expression` and '
                             '`else_expression`. ndim(condition)=' +
                             str(cond_ndim) + ', ndim(then_expression)'
                             '=' + str(expr_ndim))
        if cond_ndim > 1:
            ndim_diff = expr_ndim - cond_ndim
            cond_shape = tf.concat([tf.shape(condition), [1] * ndim_diff], axis=0)
            condition = tf.reshape(condition, cond_shape)
            expr_shape = tf.shape(then_expression)
            shape_diff = expr_shape - cond_shape
            tile_shape = tf.where(shape_diff > 0, expr_shape, tf.ones_like(expr_shape))
            condition = tf.tile(condition, tile_shape)
        x = tf.where(condition, then_expression, else_expression)
    return x

def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def generate_boxes(box_xy, box_wh, feat_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    feat_shape = tf.cast(feat_shape, box_yx.dtype)
    image_shape = tf.cast(image_shape, box_yx.dtype)

    new_shape = tf.round(image_shape * tf.reduce_min(feat_shape/image_shape))
    offset = (feat_shape - new_shape)/2.0/feat_shape
    scale = feat_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins =  box_yx - (box_hw / 2.0)
    box_maxes = box_yx + (box_hw / 2.0)
    boxes = tf.concat([box_mins[..., 0:1],
                       box_mins[..., 1:2],
                       box_maxes[..., 0:1],
                       box_maxes[..., 1:2]], axis=-1)
    boxes *= tf.concat([image_shape, image_shape], axis=-1)
    return boxes

def get_boxes_and_scores(feats, anchors, num_classes, feat_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = head(feats, anchors, num_classes,
                                                           feat_shape, image_shape)

    boxes = generate_boxes(box_xy, box_wh, feat_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])

    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    box_xy = tf.reshape(box_xy, [-1, 2])
    box_wh = tf.reshape(box_wh, [-1, 2])
    return boxes, box_scores, box_confidence, box_class_probs


class YOLOV3():
    def __init__(self, mode, input_shape, anchors, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.anchors = anchors
        self.num_anchors = len(anchors)//3
        self.num_output = self.num_anchors * (config.NUM_CLASSES+5)
        self.build(mode=mode, config=config)

        if self.mode is 'training':
            self.set_log_dir()
            self.compute_loss()
        self.eval(decode_mode=1)




    def eval(self, decode_mode=1):
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        feat_shape = tf.shape(self.outputs[0])[1:3]*32
        image_shape = tf.shape(self.input_image)[1:3]

        box_confidences = []
        box_class_probs = []
        for l in range(self.num_anchors):
            _boxes, _box_scores, _box_confidence, _box_class_probs = get_boxes_and_scores(self.outputs[l],
                                                       self.anchors[anchor_mask[l]],
                                                       self.config.NUM_CLASSES,
                                                       feat_shape,
                                                       image_shape)

            print(self.outputs[l])
            print(_box_confidence)


            boxes.append(_boxes)
            box_scores.append(_box_scores)
            box_confidences.append(tf.reshape(_box_confidence, [-1, 1]))
            box_class_probs.append(tf.reshape(_box_class_probs, [-1, self.config.NUM_CLASSES]))

        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        max_boxes_tensor = tf.constant(self.config.MAX_BOXES, dtype='int32')
        if decode_mode is 1:
            print("decode_mode 1")
            # =====================================================
            class_ids = tf.cast(tf.argmax(box_scores, axis=1), dtype=tf.int32)
            class_scores = tf.reduce_max(box_scores, axis=1)

            overthres_inds = tf.where(class_scores > 0.5)#self.config.SCORE_THRESHOLD)
            overthres_boxes = tf.gather_nd(boxes, indices=overthres_inds)
            overthres_class = tf.gather_nd(class_ids, indices=overthres_inds)
            overthres_scores = tf.gather_nd(class_scores, indices=overthres_inds)
            nms_index = tf.image.non_max_suppression(overthres_boxes, overthres_scores,
                                                     max_boxes_tensor,
                                                     iou_threshold=self.config.IOU_THRESHOLD)

            condidate_boxes = tf.gather(overthres_boxes, nms_index)
            condidate_scores = tf.gather(overthres_scores, nms_index)
            condidate_class = tf.gather(overthres_class, nms_index)

            # =====================================================
        else:
            mask = box_scores >= self.config.SCORE_THRESHOLD
            _boxes = []
            _scores = []
            _class = []
            for c in range(self.config.NUM_CLASSES):
                # TODO: use keras backend instead of tf.
                class_boxes = tf.boolean_mask(boxes, mask[:, c])
                class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
                nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores,
                                                         max_boxes_tensor, iou_threshold=self.config.IOU_THRESHOLD)

                class_boxes = tf.gather(class_boxes, nms_index)
                class_box_scores = tf.gather(class_box_scores, nms_index)
                classes = tf.ones_like(class_box_scores, 'int32') * c
                _boxes.append(class_boxes)
                _scores.append(class_box_scores)
                _class.append(classes)
            condidate_boxes = tf.concat(_boxes, axis=0)
            condidate_scores = tf.concat(_scores, axis=0)
            condidate_class = tf.concat(_class, axis=0)

        self.box_scores = condidate_scores
        self.boxes = condidate_boxes
        self.box_classes = condidate_class
        print(self.box_scores.shape)
        print(self.boxes.shape)
        print(self.box_classes.shape)


    def build(self, mode, config):
        """
        Build YOLO_V3 architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']
        if mode == 'training':
            self.is_training = True
        else:
            self.is_training = False

        print("is_training", self.is_training)

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        ss = [h // {0:32, 1:16, 2:8}[l] for l in range(3)]
        # Inputs
        self.input_image = tf.placeholder(dtype=tf.float32,
                                          shape=[None]+config.IMAGE_SHAPE.tolist(),
                                          name="input_image")

        self.y_true = [tf.placeholder(dtype=tf.float32,
                                      shape=(None, h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], 3, config.NUM_CLASSES+5)) for l in range(3)]
        print(self.y_true)

        with slim.arg_scope(training_scope(is_training=self.is_training, weight_decay=config.WEIGHT_DECAY)):
            backbone_nets = backbone_graph(self.input_image)
            layer1 =backbone_nets['Conv2d_18']
            layer2 =backbone_nets["expanded_conv_12/project"]
            layer3 =backbone_nets["expanded_conv_5/project"]
            print(layer1, layer2, layer3)
            inter_filter = 512
            x1, y1 = get_output_layer(layer1, inter_filter, self.num_output, scope='output_1')
            print(x1)
            x = concat(x1, layer2, inter_filter/2, scope='concat1')
            print(x)
            x2, y2 = get_output_layer(x, inter_filter/2, self.num_output, scope='output_2')
            print(x2)

            x = concat(x2, layer3, inter_filter/4, scope='concat2')
            print(x)
            x3, y3 = get_output_layer(x, inter_filter/4, self.num_output, scope='output_3')

            self.outputs = [y1, y2, y3]



    def compute_loss(self, ignore_thresh=0.5):
        num_classes = self.config.NUM_CLASSES
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        model_loss = 0
        box_loss = 0
        class_loss = 0
        prob_loss = 0

        input_shape = tf.cast(tf.shape(self.input_image)[1:3], dtype=tf.float32)
        grid_shapes = [self.outputs[l].get_shape().as_list()[1:3] for l in range(len(self.outputs))]
        batch_size = tf.shape(self.outputs[0])[0]
        fbatch_size = tf.cast(batch_size, dtype=tf.float32)

        self.raw_bboxes=[]
        self.raw_conf=[]
        self.raw_class=[]
        for l in range(self.num_anchors):
            object_mask = self.y_true[l][..., 4:5]
            true_calss_probs = self.y_true[l][..., 5:]

            grid, raw_pred, pred_xy, pred_wh = head(self.outputs[l], self.anchors[anchor_mask[l]],
                                                    num_classes, input_shape, calc_loss=True)

            pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

            # calc loss
            raw_true_xy = self.y_true[l][..., :2] * grid_shapes[l][::-1] - grid
            raw_true_wh = tf.log((self.y_true[l][..., 2:4] * input_shape[::-1]) / self.anchors[anchor_mask[l]].astype(np.float32))
            raw_true_wh = switch(object_mask, raw_true_wh, tf.zeros_like(raw_true_wh))

            box_loss_scale = 2 - self.y_true[l][..., 2:3] * self.y_true[l][..., 3:4]

            ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
            object_mask_bool = tf.cast(object_mask, 'bool')
            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(self.y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
                iou = box_iou(pred_box[b], true_box)
                best_iou = tf.reduce_max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, tf.cast(best_iou<0.5, dtype=true_box.dtype))
                return b+1, ignore_mask
            _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b<batch_size, loop_body, loop_vars=[0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, -1)

            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=raw_true_xy, logits=raw_pred[..., 0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - raw_pred[..., 2:4])

            sig_conf = tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=raw_pred[..., 4:5])
            confidence_loss = object_mask * sig_conf + (1-object_mask) * sig_conf * ignore_mask

            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_calss_probs, logits=raw_pred[..., 5:])


            # print(xy_loss)
            # print(wh_loss)
            # print(confidence_loss)

            xy_loss = tf.reduce_sum(xy_loss) / fbatch_size
            wh_loss = tf.reduce_sum(wh_loss) / fbatch_size
            confidence_loss = tf.reduce_sum(confidence_loss) / fbatch_size
            class_loss = tf.reduce_sum(class_loss) / fbatch_size
            model_loss += xy_loss + wh_loss + confidence_loss + class_loss

            box_loss += xy_loss + wh_loss
            class_loss += class_loss
            prob_loss += confidence_loss

        self.box_loss = box_loss
        self.class_loss = class_loss
        self.prob_loss = prob_loss
        self.model_loss = model_loss
        tf.losses.add_loss(model_loss)

        return model_loss



    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "yolo_V3_{}_*epoch*.ckpt".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

