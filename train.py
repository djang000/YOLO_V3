import os, sys
import time, colorsys
from PIL import Image, ImageFont, ImageDraw
from time import gmtime, strftime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

import libs.configs.config as config
import libs.nets.model as modellib
# from PIL import Image, ImageDraw
# import scipy.misc as sm
# from random import random
import gen_cocodb

cls_name = np.array([  'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                       'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                       'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                       'scissors', 'teddy bear', 'hair drier', 'toothbrush'])



def _get_restore_vars(scope):
    print("======== restore_variables ==============")
    print(scope)
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    variables_to_restore = []

    for var in restore_vars:
        s = var.op.name
        if s.find("Momentum") == -1:
            variables_to_restore.append(var)
    print('final')
    # for i in variables_to_restore:
    #     print(i)

    return variables_to_restore

def set_trainable(train_layers):
    # Pre-defined layer regular expressions
    print("======== set_trainable ==============")

    if train_layers == "heads":
       exclusions = ['MobilenetV2']
    else:
        return tf.global_variables()

    variables_to_train = []
    for var in tf.global_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_train.append(var)

    # print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
    # for i in variables_to_train:
    #     print(i)
    # print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")

    return variables_to_train


def print_tensors_in_checkpoint_file(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        print("tensor_name: ", key)


class CocoConfig(config.Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class_names = np.array(['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                       'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                       'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                       'scissors', 'teddy bear', 'hair drier', 'toothbrush'])

def train(train_dataset, model, config, lr, train_layers, epochs):

    """ set Solver for losses """
    global_step = slim.create_global_step()
    learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=config.LEARNING_MOMENTUM, name='Momentum')

    model_loss = tf.get_collection(tf.GraphKeys.LOSSES)
    print(model_loss)
    model_loss = tf.add_n(model_loss)
    regular_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = model_loss + regular_loss

    """ set the update operations for training """
    update_ops = []
    variables_to_train = set_trainable(train_layers)
    update_opt = optimizer.minimize(total_loss, global_step=global_step, var_list=variables_to_train)
    update_ops.append(update_opt)

    update_bns = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if len(update_bns):
        update_bn = tf.group(*update_bns)
        update_ops.append(update_bn)
    update_op = tf.group(*update_ops)

    """ set Summary and log info """
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('regular_loss', regular_loss)
    tf.summary.scalar('learning_rate', learning_rate)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model.log_dir, graph=tf.Session().graph)

    """ Set Gpu Env """
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    """ Starting Training..... """
    gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_opt)) as sess:
        sess.run(init_op)
        """ set saver for saving final model and backbone model for restore """
        # vars_restore = _get_restore_vars(None)
        # re_saver = tf.train.Saver(max_to_keep=3, var_list=vars_restore)
        saver = tf.train.Saver(max_to_keep=3)
        ckpt = tf.train.get_checkpoint_state("output/training")
        """ resotre checkpoint of Backbone network """
        if ckpt:
            lastest_ckpt = tf.train.latest_checkpoint("output/training")
            print('lastest', lastest_ckpt)
            # re_saver.restore(sess, lastest_ckpt)

        b=0 # batch index
        num_epoch = 0
        batch_size = config.BATCH_SIZE
        image_index = -1
        image_ids = np.copy(train_dataset.image_ids)
        print(batch_size)
        print("============ Start for ===================")
        try:
            while True:
                image_index = (image_index + 1) % len(image_ids)
                # shuffle images if at the start of an epoch.
                if image_index == 0:
                    np.random.shuffle(image_ids)
                    num_epoch +=1

                # Get gt_boxes and gt_masks for image.
                image_id = image_ids[image_index]
                # image_id = 47082
                image, gt_boxes = train_dataset.load_gt(train_dataset, image_id, config)
                if len(gt_boxes) is 0:
                    print('gt_boxes of [imageid : %d] is not exist !!!'%image_id)
                    continue

                # Init batch arrays
                if b == 0:
                    batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 5), dtype=np.int32)

                # If more instances than fits in the array, sub-sample from them.
                if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                    print("Gt is too much!!")
                    ids = np.random.choice(
                        np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                    gt_boxes = gt_boxes[ids]

                # Add to batch
                batch_images[b] = image.astype(np.float32)
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                b += 1

                # Batch full?
                if b >= batch_size:
                    y_true = generate_gt_labels(batch_gt_boxes, config.IMAGE_SHAPE[:2], anchors, config.NUM_CLASSES)
                    feed_dict = {model.input_image: batch_images, model.y_true[0]: y_true[0],
                                 model.y_true[1]: y_true[1], model.y_true[2]: y_true[2],
                                 learning_rate:lr}
                    _, loss, class_loss, box_loss, prob_loss, r_loss, current_step, summary = sess.run([update_op, total_loss, model.class_loss,
                                                                                                        model.box_loss, model.prob_loss, regular_loss,
                                                                                                        global_step, summary_op], feed_dict=feed_dict)

                    print ("""iter %d / %d : total-loss %.4f (c : %.4f, b : %.4f, score : %.4f, reglur : %.4f)""" %
                           (current_step, num_epoch, loss, class_loss, box_loss, prob_loss, r_loss))

                    if np.isnan(loss) or np.isinf(loss):
                        print('isnan or isinf', loss)
                        raise
                    if current_step % 1000 == 0:
                        # write summary
                        # summary = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary, current_step)
                        summary_writer.flush()

                    if current_step % 3000 == 0:
                        # Save a checkpoint
                        save_path = 'output/training/yoloV3.ckpt'
                        saver.save(sess, save_path, global_step=current_step)

                    if num_epoch > epochs:
                        print("num epoch : %d and training End!!!" % num_epoch)
                        break

                    b = 0
                    # break
        except Exception as ex:
            print('Error occured!!!! => ', ex)
            # 40040
        finally:
            print("Final!!")
            saver.save(sess, 'output/models/yoloV3_final.ckpt', write_meta_graph=False)
            sess.close()

def generate_gt_labels(gt_boxes, input_shape, anchors, num_classes):
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    true_boxes = np.array(gt_boxes, dtype=np.float32)
    input_shape = np.array(input_shape, dtype=np.int32)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]

    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]

    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), num_classes + 5), dtype=np.float32)
              for l in range(num_layers)]
    y_true[0][:, :, :, :, 5] = 1
    y_true[1][:, :, :, :, 5] = 1
    y_true[2][:, :, :, :, 5] = 1

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1
                    y_true[l][b, j, i, k, 5] = 0

    return y_true




def draw_pred_boxes(image, boxes, y_true):
    boxes = np.squeeze(boxes, 0)
    print(boxes.shape)
    image = np.asarray(image * 255.0, dtype="uint8")
    print('Found {} boxes for {}'.format(len(boxes), 'img'))

    image = Image.fromarray(image)
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    hsv_tuples = [(x / 81., 1., 1.) for x in range(81)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for i in range(len(boxes)):
        c = boxes[i, 4].astype('int32')
        print(c)
        label = '{} :{}'.format(c, class_names[c])
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = boxes[i, :4]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    image.show()


if __name__ == "__main__":
    mode = "training"
    print("mode ==> %%s!!", mode)

    # Root directory of the project
    ROOT_DIR = os.getcwd()
    # Directory to save logs and model checkpoints, if not provided
    # through the command line argument --logs
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "output/logs")

    dataset_path = os.path.join(ROOT_DIR, 'data/coco')
    config = CocoConfig()

    print('dataset_path: ', dataset_path)
    coco_train = gen_cocodb.CocoDataSet()
    coco_train.load_coco(dataset_path, "train", year="2014", auto_download=False)
    coco_train.load_coco(dataset_path, "valminusminival", year="2014", auto_download=False)
    coco_train.prepare()
    print("size", len(coco_train.image_ids), len(coco_train.image_ids) * 160/config.BATCH_SIZE)

    with open('data/yolo_anchors.txt') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    print(anchors.shape, anchors, config.NUM_CLASSES)

    with tf.device("/GPU:0"):
        model = modellib.YOLOV3('training', (416, 416), anchors, config, model_dir=DEFAULT_LOGS_DIR)

    # iter : 9777440
    train(coco_train, model, config, lr=config.LEARNING_RATE*0.01, train_layers="all", epochs=1000)



