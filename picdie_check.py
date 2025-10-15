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
		self.tobepath = fn
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


def drawtangle2(box,cls,cimg,color,tscore):


	ymin = int(box[0])
	xmin = int(box[1])
	ymax = int(box[2])
	xmax = int(box[3])

	cv2.rectangle(cimg,(xmin-1,ymin-1),(xmax+1,ymax+1),(0,0,255),2)

	ref_dict = {}
	ref_dict[0] = 'ng'
	ref_dict[1] = 'pd'
	ref_dict[2] = 'wd'

	cv2.putText(cimg,ref_dict[int(cls)],(xmin+30,ymin+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

	if xmax < 200:
		cv2.putText(cimg,str(tscore)+','+str(xmax-xmin)+','+str(ymax-ymin),(xmin+60,ymin+60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)
	else:
		cv2.putText(cimg,str(tscore)+','+str(xmax-xmin)+','+str(ymax-ymin),(xmin-120,ymin+60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)


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
		gamma = 0.9
		if '_WAVEGUIDE_' in param.tobepath.upper():
			param.score = 0.4
			gamma = 1.1

		if '_METALTRACE_' in param.tobepath.upper() or '_MODULATOR_' in param.tobepath.upper():
			gamma = 1.1

		newimgpath = param.tobepath.replace(".jpg","_0.jpg")

		cimg = cv2.imread(param.tobepath,cv2.IMREAD_COLOR)
		cimg = cv2.resize(cimg,(WIDTH,HIGH))
		cimg = CLAHE(cimg,2.1,16)
		cimg = gamma_correction_lab(cimg,gamma)
		cv2.imwrite(newimgpath,cimg)


		# input_image_size = (HIGH, WIDTH)
		img = tf.io.read_file(newimgpath)
		img_tensor = tf.io.decode_image(img, channels=3)
		# img_tensor = tf.image.resize(img_tensor,input_image_size)
		img_tensor = tf.expand_dims(img_tensor, axis=0)
		img_tensor = tf.cast(img_tensor, dtype = tf.uint8)
		
		ngsubfix = '.jpg'
		cimg = cv2.imread(newimgpath,cv2.IMREAD_COLOR)
		output_dict = {}
		output_dict = param.model_fn(img_tensor)
		for i in range(100):
			if float(output_dict['detection_scores'][0][i]) >= param.score:
				clsidx = int(output_dict['detection_classes'][0][i]) - 1
				if clsidx == 0:
					ngsubfix = '_xxx.jpg'
				color = param.colors[clsidx%30]
				tscore = round(100.0*float(output_dict['detection_scores'][0][i]),2)
				bbox = output_dict['detection_boxes'][0][i]
				drawtangle2(bbox,clsidx,cimg,color,tscore)
		cv2.imwrite(param.tobepath.replace("test","testout").replace('.jpg',ngsubfix),cimg,[cv2.IMWRITE_JPEG_QUALITY,100])

		try:
			os.remove(newimgpath)
		except:
			print('fail to remove file')
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
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpuid', type=str, help='gpu device id', required=True)
	# parser.add_argument('--runid', type=int, help='run id', required=True)
	args = parser.parse_args()
	# print('python picdie_check.py  --gpuid  '+args.gpuid+'  --runid  '+str(args.runid))
	print('python picdie_check.py  --gpuid  '+args.gpuid)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
	gpus = tf.config.list_physical_devices('GPU')
	tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3*1024)])
	logical_gpus = tf.config.list_logical_devices('GPU')
	print(logical_gpus)

	with tf.device('/device:GPU:0'):
		try:
			paramlist = []
			score = 0.5
			colors = random_colors(30)

			print("loading test data..............")

			data_root = pathlib.Path('./mydata/trainsrcdata/PICDIE/test7')
			all_image_paths = list(data_root.glob('*'))
			all_image_paths = [str(path) for path in all_image_paths]

			print("loading model files............")

			export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_791'#491
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_sys01_785'#494
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_sys01_789x'#455
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_786'#451
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_sys01_795'#455
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_sys01_791'#448
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_789'#452
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_sys01_788'#454
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_sys01_802'#422
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_802'# 427
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_sys01_799x'#428
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_804'#435

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B71_807'#446
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B71_sys01_806'#439
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B71_805'#430
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B71_808'#446

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B51_sys01_786'#449
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B52_765'#450
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B51_787'#453

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
