

import os
import sys
# import warnings
# warnings.filterwarnings("ignore")

import logging
logging.getLogger('absl').setLevel('ERROR')

import tensorflow as tf
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import io
# import matplotlib
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)

import pathlib
import cv2
import copy
import uuid

import math

import colorsys
import random

from PIL import Image
# from six import BytesIO

# import orbit
# import tensorflow_models as tfm


# from official.core import exp_factory
# from official.core import config_definitions as cfg
# from official.vision.serving import export_saved_model_lib
# from official.vision.ops.preprocess_ops import normalize_image
# from official.vision.ops.preprocess_ops import resize_image
# from official.vision.utils.object_detection import visualization_utils
# from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder


from absl import app
from absl import flags
from absl import logging
# import gin



# from official.common import distribute_utils
# from official.common import flags as tfm_flags
# from official.core import task_factory
# from official.core import train_lib
# from official.core import train_utils
# from official.modeling import performance
# from official.vision import registry_imports  # pylint: disable=unused-import
# from official.vision.utils import summary_manager


import dataclasses
from typing import Optional, List, Sequence, Union

# from official.core import config_definitions as cfg
# from official.core import exp_factory
# from official.modeling import hyperparams
# from official.modeling import optimization
# from official.modeling.hyperparams import base_config
# from official.vision.configs import common
# from official.vision.configs import decoders
# from official.vision.configs import backbones

# from official.vision.configs import retinanet

HIGH = 640
WIDTH = 640


def random_colors(N, bright=True):
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors


def drawtangle(box,cls,maskts,cimg,color):
	ymin = int(box[0])
	xmin = int(box[1])
	ymax = int(box[2])
	xmax = int(box[3])

	cx = int((xmin+xmax)/2)
	cy = int((ymin+ymax)/2)

	cv2.rectangle(cimg,(xmin,ymin),(xmax,ymax),(0,255,0),2)
	


	mask = maskts.numpy()
	mask = cv2.resize(mask, ((xmax-xmin), (ymax-ymin)))
	mask = (mask*255).astype("uint8")
	# print('mask:')
	# print(mask)


	rnnmask = np.zeros((640,640), dtype="uint8")
	rnnmask[ymin:ymax,xmin:xmax] = mask

	alpha=0.5
	for c in range(3):
		cimg[:, :, c] = np.where(rnnmask > 50,cimg[:, :, c] *(1 - alpha) + alpha * color[c] * 255,cimg[:, :, c])

	cv2.putText(cimg,str(int(cls)),(xmin+10,ymin+20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),3)

	# im = Image.fromarray(rnnmask)
	# im.save('./mydata/mask'+str(i)+'.jpg')

export_dir = './mydata/exported_model_2500_mzurich'


imported = tf.saved_model.load(export_dir)
model_fn = imported.signatures['serving_default']

colors = random_colors(10)

print(colors)

input_image_size = (HIGH, WIDTH)
score = 0.85

data_root = pathlib.Path('./mydata/check_piczu')
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]
for fn in all_image_paths:
	if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
		img = tf.io.read_file(fn)
		img_tensor = tf.io.decode_image(img, channels=3)
		img_tensor = tf.image.resize(img_tensor,input_image_size)
		img_tensor = tf.expand_dims(img_tensor, axis=0)
		img_tensor = tf.cast(img_tensor, dtype = tf.uint8)

		output_dict = model_fn(img_tensor)
		# print(output_dict)
		
		cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
		cimg = cv2.resize(cimg,(WIDTH,HIGH))
		for i in range(100):
			if float(output_dict['detection_scores'][0][i]) >= score:
				color = colors[i%10]
				drawtangle(output_dict['detection_boxes'][0][i],output_dict['detection_classes'][0][i],output_dict['detection_masks'][0][i],cimg,color)

		cimg = cv2.resize(cimg,(640,480))
		cv2.imwrite(fn.replace('.jp','_2000_'+str(int(score*100.0))+'.jp').replace('check_pic','checked_pic'),cimg)



