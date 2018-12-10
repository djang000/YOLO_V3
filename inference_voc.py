import colorsys
import os
import random
from timeit import default_timer as timer
import scipy.misc

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf

import libs.configs.config as config
import libs.nets.model as modellib

class_names = np.array(['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                        'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor'])

class CocoConfig(config.Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "voc"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # COCO has 80 classes

def detect_img(input_image, model):
    start = timer()

    # Generate colors for drawing bounding boxes.
    num_class = len(class_names)
    hsv_tuples = [(x / num_class, 1., 1.) for x in range(num_class)]

    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    # Performing post-processing on CPU: loop-intensive, usually more efficient.
    with tf.Session() as sess:
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state('output/training')
        """ resotre checkpoint of Backbone network """
        if ckpt is not None:
            ckpt_path = tf.train.latest_checkpoint('output/training')
            saver.restore(sess, ckpt_path)
        else:
            ckpt_path = 'output/models.final.ckpt'
            saver.restore(sess, ckpt_path)
        print('ckpt_path', ckpt_path)
        #======================================================================
        out_boxes, out_scores, out_classes = sess.run([model.boxes, model.box_scores, model.box_classes],
                                         feed_dict={ model.input_image: input_image })
        print(out_boxes.shape, out_scores.shape, out_classes.shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        image = np.squeeze(input_image, 0)
        image = image*255.0

        image = Image.fromarray(image.astype('uint8'))
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i in range(len(out_classes)):
            c = out_classes[i]
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.5f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
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

        end = timer()
        print("processing time : %f sec"%(end - start))
        image.show()
        sess.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = CocoConfig()

    image = Image.open('images/test_3.jpg')
    print(image.size[0],image.size[1])
    image = scipy.misc.imresize(image, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, 0)

    with open('data/yolo_anchors.txt') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    print(anchors.shape, anchors, config.NUM_CLASSES)

    with tf.device("/CPU:0"):
        yolo_model = modellib.YOLOV3('inference', (416, 416), anchors, config, model_dir=None)

    detect_img(image, yolo_model)