import argparse
import os
import traceback
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

# from multiprocessing import Lock
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED
import threading
import time


class ProcParam:
	def __init__(self,lock,fn,model_fn,model_fn2,score,colors):
		self.lock = lock
		self.tobepath = fn.lower()
		self.model_fn = model_fn
		self.model_fn2 = model_fn2
		self.score = score
		self.colors = colors

wblock = threading.Lock()
HIGH = 1280
WIDTH = 1280


def random_colors(N, bright=True):
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	# random.shuffle(colors)
	retcolors = []
	retcolors.append(colors[0])
	retcolors.append(colors[10])
	retcolors.append(colors[20])
	retcolors.append(colors[28])
	retcolors.append(colors[5])
	retcolors.append(colors[15])
	retcolors.append(colors[25])
	for cc in colors:
		retcolors.append(cc)
	return retcolors


def drawtangle2(output_dict,param,cimg):

	ngjs = '.json'
	ngsubfix = '.jpg'
	for i in range(100):
		if float(output_dict['detection_scores'][0][i]) >= param.score:
			clsidx = int(output_dict['detection_classes'][0][i]) - 1
			if clsidx == 0:
				ngjs = '_X.json'
				ngsubfix = '_X.jpg'


	ref_dict = {}
	ref_dict[0] = 'ng'
	ref_dict[1] = 'pd'
	ref_dict[2] = 'wd'

	fns = param.tobepath.split('\\')
	imgname = (fns[len(fns)-1]).lower().replace('.jpg',ngsubfix)
	jsonname = param.tobepath.lower().replace("test","testout").replace('.jpg',ngjs).replace('.JPG',ngjs)

	nfo = open(jsonname,'w+')

	newline = """{
  "version": "5.4.1",
  "flags": {},
  "shapes": ["""

	nfo.write(newline)

	writed = 0
	for i in range(100):
		clsidx = int(output_dict['detection_classes'][0][i]) - 1
		if float(output_dict['detection_scores'][0][i]) >= param.score or (clsidx == 1 and float(output_dict['detection_scores'][0][i]) > 0.4):

			box = output_dict['detection_boxes'][0][i]
			ymin = str(float(box[0]))
			xmin = str(float(box[1]))
			ymax = str(float(box[2]))
			xmax = str(float(box[3]))

			yymin = int(box[0]*2048/1280)
			xxmin = int(box[1]*2448/1280)
			yymax = int(box[2]*2048/1280)
			xxmax = int(box[3]*2448/1280)
			xwidth = int(float(xxmax-xxmin)/120.0*40.0)
			yheight = int(float(yymax-yymin)/120.0*40.0)


			# if clsidx == 0:
			# 	continue

			if clsidx == 1 and yymax < 200:
				continue

			if clsidx == 1 and yheight < 35:
				continue
			
			lab = ref_dict[int(clsidx)]

			newline="""
    {
      "label": "xxxx",
      "points": [
        [
          xmin,
          ymin
        ],
        [
          xmax,
          ymax
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    }"""
			newline = newline.replace('xxxx',lab).replace('xmin',xmin).replace('ymin',ymin).replace('xmax',xmax).replace('ymax',ymax)
			tscore = round(100.0*float(output_dict['detection_scores'][0][i]),2)

			if lab == 'pd' or lab == 'wd':
				if tscore > 90:
					if writed != 0:
						nfo.write(',')
					nfo.write(newline)
					writed = 1
			else:
				if writed != 0:
					nfo.write(',')
				nfo.write(newline)
				writed = 1
	newline="""
  ],
  "imagePath": "imgname",
  "imageData": null,
  "imageHeight": 1280,
  "imageWidth": 1280
}"""
	
	newline = newline.replace('imgname',imgname)
	nfo.write(newline)

	nfo.close()

	cv2.imwrite(param.tobepath.replace("test","testout").replace('.jpg',ngsubfix).replace('.JPG',ngsubfix),cimg,[cv2.IMWRITE_JPEG_QUALITY,100])



def gamma_correction(img, gamma=1.0):
	img_normalized = img.astype(np.float32) / 255.0
	corrected = np.power(img_normalized, gamma)
	corrected = (corrected * 255).astype(np.uint8)
	return corrected

def gamma_correction_lab(img, gamma=1.0):
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	l_corrected = gamma_correction(l, gamma)
	lab_corrected = cv2.merge((l_corrected, a, b))
	return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


def CLAHE(cimg,clipLimit,tSize):
	lab = cv2.cvtColor(cimg,cv2.COLOR_BGR2LAB)
	l,a,b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit,tileGridSize=(tSize,tSize))
	l_clahe = clahe.apply(l)
	lab_clahe = cv2.merge((l_clahe,a,b))
	return cv2.cvtColor(lab_clahe,cv2.COLOR_LAB2BGR)



def run_model(param):
	print("   \r\n")
	print(param.tobepath)

	try:
		upperpath = param.tobepath.upper()
		gamma = 0.9
		if '_WAVEGUIDE_' in upperpath:
			param.score = 0.4
			gamma = 0.7

		# newimgpath = param.tobepath.replace(".jpg","_0.jpg")

		cimg = cv2.imread(param.tobepath,cv2.IMREAD_COLOR)
		hg,wd,ch = cimg.shape
		if hg != 1280:
			cimg = cv2.resize(cimg,(WIDTH,HIGH))
			cimg = CLAHE(cimg,2.1,16)
			cimg = gamma_correction_lab(cimg,gamma)
		# cv2.imwrite(newimgpath,cimg)


		# input_image_size = (HIGH, WIDTH)
		# img = tf.io.read_file(newimgpath)

		rgbimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2RGB)
		img_tensor = tf.convert_to_tensor(rgbimg,dtype=tf.float32)


		# img_tensor = tf.io.decode_image(img, channels=3)
		# img_tensor = tf.image.resize(img_tensor,input_image_size)

		img_tensor = tf.expand_dims(img_tensor, axis=0)
		img_tensor = tf.cast(img_tensor, dtype = tf.uint8)
		

		# cimg = cv2.imread(newimgpath,cv2.IMREAD_COLOR)
		output_dict = param.model_fn(img_tensor)
		drawtangle2(output_dict,param,cimg)
			
		# try:
		# 	os.remove(newimgpath)
		# except:
		# 	print('fail to remove file')

	except:
		exception_message = sys.exc_info()[1]
		print(str(exception_message))
		traceback.print_exc()


def getRunID(filepath):
    try:
        fps = filepath.split("\\")
        if(len(fps) < 2):
            fps = filepath.split("/")
        ids = fps[len(fps)-1].split("_")
        idx = int(ids[0])%4
    except:
        print('exception file path in runid:'+filepath)
        return 0
    return idx

def MainLoop():
	print('picdie_check start...........')

	parser = argparse.ArgumentParser()
	parser.add_argument('--gpuid', type=str, help='gpu device id', required=True)
	# parser.add_argument('--runid', type=int, help='run id', required=True)
	args = parser.parse_args()
	# print('python picdie_check.py  --gpuid  '+args.gpuid+'  --runid  '+str(args.runid))
	print('python picdie_marker.py  --gpuid  '+args.gpuid)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
	gpus = tf.config.list_physical_devices('GPU')
	tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6*1024)])
	logical_gpus = tf.config.list_logical_devices('GPU')
	print(logical_gpus)

	with tf.device('/device:GPU:0'):
		try:
			paramlist = []
			score = 0.5
			colors = random_colors(30)

			print("loading test data..............")

			data_root = pathlib.Path('./mydata/trainsrcdata/PICDIE/test1vgf')
			all_image_paths = list(data_root.glob('*'))
			all_image_paths = [str(path) for path in all_image_paths]

			print("loading model files............")

			export_dir = './mydata/trainmodels/ORION/exported_model_OR_8542xxxxx0'
			
			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8495xxxxx'

			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8528xxxxx'

			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8526xxxxx'

			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8529xxxxx' 

			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_850xxxx'

			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8487xxxx'

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8364xxx'#


			imported = tf.saved_model.load(export_dir)
			model_fn = imported.signatures['serving_default']

			# # export_dir2 = './mydata/trainmodels/EMLCOCSWD/exported_model64_2760_906'
			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B21_56_0617'
			# # export_dir2 = './mydata/trainmodels/EMLCOCSWD/exported_model75_9006'
			# imported2 = tf.saved_model.load(export_dir2)
			# model_fn2 = imported2.signatures['serving_default']

			for fn in all_image_paths:
				if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
					param = ProcParam(wblock,fn,model_fn,None,score,colors)
					paramlist.append(param)

			print("get test data count: "+str(len(paramlist)))

			for param in paramlist:
				run_model(param)

		except:
			print('run into exception0.......')
			exception_message = sys.exc_info()[1]
			print(str(exception_message))

if __name__ == "__main__":
	MainLoop()
