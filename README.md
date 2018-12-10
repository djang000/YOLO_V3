# YOLO_V3

### Installation
if you want to use pretrained models, then all you need to do is:
```sh
git clone https://oss.navercorp.com/deep-purple/YOLO_V3.git
```

if you also want to train new modes, you will need the MS-COCO, VOC or other natural images for training files and Mobienet_V2-1.0 wegihts by running.

you can download Mobienet_V2 weight from below website
```sh
https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
```

### Usage

Following are examples of how the scripts in this repo can be used. 

- Train using your dataset
you must use train.py for COCO dataet. but if you want to train for VOC dataet, you can use train_voc.py

Example usage:

```sh
python train.py
```

- Inference

Generate object detection with pre-trained model date.

you use inference.py for COCO data. Also you can check inference with trained model data for VOC dataset using inderence_voc.py

Example usage:

```sh
python inference.py
```	

