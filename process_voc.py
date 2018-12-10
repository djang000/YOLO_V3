import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.io
import skimage.color
from collections import namedtuple
from tqdm import tqdm
from glob import glob
import lxml.etree
import os, random

Label   = namedtuple('Label',   ['name', 'color'])
Sample  = namedtuple('Sample',  ['filename', 'boxes'])
#-------------------------------------------------------------------------------
def rgb2bgr(tpl):
    """
    Convert RGB color tuple to BGR
    """
    return (tpl[2], tpl[1], tpl[0])

label_defs = [
    Label('background',  rgb2bgr((29,   36, 118))),
    Label('aeroplane',   rgb2bgr((0,     0,   0))),
    Label('bicycle',     rgb2bgr((111,  74,   0))),
    Label('bird',        rgb2bgr(( 81,   0,  81))),
    Label('boat',        rgb2bgr((128,  64, 128))),
    Label('bottle',      rgb2bgr((244,  35, 232))),
    Label('bus',         rgb2bgr((230, 150, 140))),
    Label('car',         rgb2bgr(( 70,  70,  70))),
    Label('cat',         rgb2bgr((102, 102, 156))),
    Label('chair',       rgb2bgr((190, 153, 153))),
    Label('cow',         rgb2bgr((150, 120,  90))),
    Label('diningtable', rgb2bgr((153, 153, 153))),
    Label('dog',         rgb2bgr((250, 170,  30))),
    Label('horse',       rgb2bgr((220, 220,   0))),
    Label('motorbike',   rgb2bgr((107, 142,  35))),
    Label('person',      rgb2bgr(( 52, 151,  52))),
    Label('pottedplant', rgb2bgr(( 70, 130, 180))),
    Label('sheep',       rgb2bgr((220,  20,  60))),
    Label('sofa',        rgb2bgr((  0,   0, 142))),
    Label('train',       rgb2bgr((  0,   0, 230))),
    Label('tvmonitor',   rgb2bgr((119,  11,  32)))]

#-------------------------------------------------------------------------------
class PascalVOCSource:
    #---------------------------------------------------------------------------
    def __init__(self):
        self.num_classes   = len(label_defs)
        self.colors        = {l.name: l.color for l in label_defs}
        self.lid2name      = {i: l.name for i, l in enumerate(label_defs)}
        self.lname2id      = {l.name: i for i, l in enumerate(label_defs)}
        self.num_train     = 0
        self.num_valid     = 0
        self.num_test      = 0
        self.train_samples = []
        self.valid_samples = []
        self.test_samples  = []

    # ---------------------------------------------------------------------------
    def __build_annotation_list(self, root, dataset_type):
        """
        Build a list of samples for the VOC dataset (either trainval or test)
        """
        annot_root = root + '/Annotations/'
        annot_files = []
        with open(root + '/ImageSets/Main/' + dataset_type + '.txt') as f:
            for line in f:
                annot_file = annot_root + line.strip() + '.xml'
                if os.path.exists(annot_file):
                    annot_files.append(annot_file)
        return annot_files

    # ---------------------------------------------------------------------------
    def __build_sample_list(self, root, annot_files, dataset_name):
        """
        Build a list of samples for the VOC dataset (either trainval or test)
        """
        image_root = root + '/JPEGImages/'
        samples = []

        # -----------------------------------------------------------------------
        # Process each annotated sample
        # -----------------------------------------------------------------------
        for fn in tqdm(annot_files, desc=dataset_name, unit='samples'):
            with open(fn, 'r') as f:
                doc = lxml.etree.parse(f)
                filename = image_root + doc.xpath('/annotation/filename')[0].text

                # ---------------------------------------------------------------
                # Get the file dimensions
                # ---------------------------------------------------------------
                if not os.path.exists(filename):
                    continue

                # ---------------------------------------------------------------
                # Get boxes for all the objects
                # ---------------------------------------------------------------
                boxes = []
                objects = doc.xpath('/annotation/object')
                for obj in objects:
                    # -----------------------------------------------------------
                    # Get the properties of the box and convert them to the
                    # proportional terms
                    # -----------------------------------------------------------
                    label = obj.xpath('name')[0].text
                    xmin = int(float(obj.xpath('bndbox/xmin')[0].text))
                    xmax = int(float(obj.xpath('bndbox/xmax')[0].text))
                    ymin = int(float(obj.xpath('bndbox/ymin')[0].text))
                    ymax = int(float(obj.xpath('bndbox/ymax')[0].text))
                    boxes.append([xmin, ymin, xmax, ymax, self.lname2id[label]])
                if not boxes:
                    continue
                sample = Sample(filename, boxes)
                samples.append(sample)

        return samples

    def load_trainval_data(self, data_dir, valid_fraction=0.025):
        """
        Load the training and validation data
        :param data_dir:       the directory where the dataset's file are stored
        :param valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """

        #-----------------------------------------------------------------------
        # Process the samples defined in the relevant file lists
        #-----------------------------------------------------------------------
        train_annot = []
        train_samples = []
        for vocid in ['VOC2007', 'VOC2012']:
            root = data_dir + '/trainval/VOCdevkit/'+vocid
            name = 'trainval_'+vocid
            annot = self.__build_annotation_list(root, 'trainval')
            train_annot += annot
            train_samples += self.__build_sample_list(root, annot, name)

        root = data_dir + '/test/VOCdevkit/VOC2007'
        annot = self.__build_annotation_list(root, 'test')
        train_samples += self.__build_sample_list(root, annot, 'test_VOC2007')

        #-----------------------------------------------------------------------
        # We have some 5.5k annotated samples that are not on these lists, so
        # we can use them for validation
        #-----------------------------------------------------------------------
        root = data_dir + '/trainval/VOCdevkit/VOC2012'
        all_annot = set(glob(root + '/Annotations/*.xml'))
        valid_annot = all_annot - set(train_annot)
        valid_samples = self.__build_sample_list(root, valid_annot,
                                                 'valid_VOC2012')

        #-----------------------------------------------------------------------
        # Final set up and sanity check
        #-----------------------------------------------------------------------
        self.valid_samples = valid_samples
        self.train_samples = train_samples

        if len(self.train_samples) == 0:
            raise RuntimeError('No training samples found in ' + data_dir)

        if valid_fraction > 0:
            if len(self.valid_samples) == 0:
                raise RuntimeError('No validation samples found in ' + data_dir)

        self.num_train = len(self.train_samples)
        self.num_valid = len(self.valid_samples)


    #---------------------------------------------------------------------------
    def load_test_data(self, data_dir):
        """
        Load the test data
        :param data_dir: the directory where the dataset's file are stored
        """
        root = data_dir + '/test/VOCdevkit/VOC2007'
        annot = self.__build_annotation_list(root, 'test')
        self.test_samples  = self.__build_sample_list(root, annot,
                                                      'test_VOC2007')

        if len(self.test_samples) == 0:
            raise RuntimeError('No testing samples found in ' + data_dir)

        self.num_test  = len(self.test_samples)


    def __BrightnessTransform(self, data, delta=32):
        data = data.astype(np.float32)
        delta = random.randint(-delta, delta)
        data += delta
        data[data > 255] = 255
        data[data < 0] = 0
        data = data.astype(np.uint8)
        return data

    def __ContrastTransform(self, data, lower=0.5, upper=1.5):
        data = data.astype(np.float32)
        delta = random.uniform(lower, upper)
        data *= delta
        data[data>255] = 255
        data[data<0] = 0
        data = data.astype(np.uint8)
        return data

    def __preprocessing(self, image, boxes, config):
        h, w = image.shape[:2]
        scale_w = float(config.IMAGE_MAX_DIM) / float(w)
        scale_h = float(config.IMAGE_MAX_DIM) / float(h)
        image = scipy.misc.imresize(image, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM))

        boxes[:, 0] = boxes[:, 0] * scale_w
        boxes[:, 1] = boxes[:, 1] * scale_h
        boxes[:, 2] = boxes[:, 2] * scale_w
        boxes[:, 3] = boxes[:, 3] * scale_h

        # Random horizontal flips.
        if random.randint(0, 1):
            image = np.fliplr(image)
            boxes[:, [0, 2]] = image.shape[1] - boxes[:, [2, 0]]

        # Random Brightness Transform.
        if random.randint(0, 1):
            image = self.__BrightnessTransform(image)

        # Random Contrast Transform.
        if random.randint(0, 1):
            image = self.__ContrastTransform(image)


        gt_boxes = []
        for i in range(len(boxes)):
            w, h = boxes[i, 2:4] - boxes[i, 0:2]
            if w > 2.0 and h > 2.0:
                gt_boxes.append(boxes[i])

        # image = (np.array(image)/255.0 - 0.5)*2.0
        image = np.array(image)/255.0
        gt_boxes = np.asarray(gt_boxes)
        return image, gt_boxes

    def load_sample(self, sample, config):
        # print(sample)
        files = sample.filename
        bboxes = np.asarray(sample.boxes)
        # Load image
        image = skimage.io.imread(files)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        image, gt_boxes = self.__preprocessing(image, bboxes, config)

        return image, gt_boxes





if __name__ == "__main__":
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and model checkpoints, if not provided
    # through the command line argument --logs
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "output/logs")

    dataset_path = os.path.join(ROOT_DIR, 'data/VOC')

    print('dataset_path: ', dataset_path)
    voc_source = PascalVOCSource()
    voc_source.load_trainval_data(dataset_path)
