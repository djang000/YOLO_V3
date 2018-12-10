import os
import numpy as np
import tensorflow as tf

import libs.configs.config as config
import libs.nets.model as modellib
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference



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
    NUM_CLASSES = 1 + 80  # COCO has 80 classes



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = CocoConfig()

    INPUT_POINT = "input_image"
    PB_BOXES = "eval/condidate_boxes"
    PB_SCORES = "eval/condidate_scores"
    PB_CLASSES = "eval/condidate_class"
    OUTPUT_PB_FILENAME = "output/models/YOLOV3_model_quant.pb"


    with open('data/yolo_anchors.txt') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    print(anchors.shape, anchors, config.NUM_CLASSES)

    graph = tf.Graph()
    with graph.as_default():
        yolo_model = modellib.YOLOV3('inference', (416, 416), anchors, config, model_dir=None)

        # Training model
        saver = tf.train.Saver()
        with tf.Session(graph=graph) as sess:
            save_path = "output/models/yoloV3_final.ckpt"
            saver.restore(sess, save_path)

            tf.contrib.quantize.create_eval_graph(input_graph=tf.get_default_graph())



            constant_graph = convert_variables_to_constants(sess,
                                                            sess.graph_def,
                                                            [INPUT_POINT, PB_BOXES, PB_SCORES, PB_CLASSES])

            optimized_constant_graph = optimize_for_inference(constant_graph,
                                                              [INPUT_POINT],
                                                              [PB_BOXES, PB_SCORES, PB_CLASSES],
                                                              tf.float32.as_datatype_enum)

            # Generate PB file and we also generate text file for debug on graph
            tf.train.write_graph(optimized_constant_graph, '.', OUTPUT_PB_FILENAME, as_text=False)
            tf.train.write_graph(optimized_constant_graph, '.', OUTPUT_PB_FILENAME + ".txt", as_text=True)

        # print PB file size
        filesize = os.path.getsize(OUTPUT_PB_FILENAME)
        filesize_mb = filesize / 1024 / 1024
        print(str(round(filesize_mb, 3)) + " MB")

        sess.close()

