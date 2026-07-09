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

def drawtangle2(box,cls,cimg,color,tscore,param):

	ymin = int(box[0]*2048/1280)
	xmin = int(box[1]*2448/1280)
	ymax = int(box[2]*2048/1280)
	xmax = int(box[3]*2448/1280)

	xwidth = int(float(xmax-xmin)/120.0*40.0)
	yheight = int(float(ymax-ymin)/120.0*40.0)

	left = xmin-10
	right = xmax+10
	up = ymin-10
	down = ymax+10
	if left < 0:
		left = 0
	if right > 2448:
		right = 2448
	if up < 0:
		up = 0
	if down > 2048:
		down = 2048

	cv2.rectangle(cimg,(left,up),(right,down),(0,0,255),1)

	ref_dict = {}
	ref_dict[0] = 'NG'
	ref_dict[1] = 'PD'
	ref_dict[2] = 'WD'

	xmark = xmin+10
	ymark = ymax+42

	if xmax > 2200:
		xmark = xmin-100
	if ymax > 1900:
		ymark = ymin-100

	cv2.putText(cimg,ref_dict[int(cls)]+','+str(xwidth)+','+str(yheight)+':'+str(tscore),(xmark,ymark),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
	# cv2.putText(cimg,str(tscore),(xmark,ymark+35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

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

		# if '_METALTRACE_' in param.tobepath.upper() or '_MODULATOR_' in param.tobepath.upper():
		# 	gamma = 1.1

		# newimgpath = param.tobepath.replace(".jpg","_0.jpg")

		cimgx = cv2.imread(param.tobepath,cv2.IMREAD_COLOR)
		hg,wd,ch = cimgx.shape
		cimg = cv2.resize(cimgx,(WIDTH,HIGH))
		
		if hg != 1280:
			cimg = CLAHE(cimg,2.1,16)
			cimg = gamma_correction_lab(cimg,gamma)
		else:
			cimgx = cv2.resize(cimgx,(2448,2048))

		#cv2.imwrite(newimgpath,cimg)
		#img = tf.io.read_file(newimgpath)
		#img_tensor = tf.io.decode_image(img, channels=3)

		rgbimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2RGB)
		img_tensor = tf.convert_to_tensor(rgbimg,dtype=tf.float32)

		# input_image_size = (HIGH, WIDTH)
		# img_tensor = tf.image.resize(img_tensor,input_image_size)
		
		img_tensor = tf.expand_dims(img_tensor, axis=0)
		img_tensor = tf.cast(img_tensor, dtype = tf.uint8)
		
		ngsubfix = '.jpg'

		output_dict = {}
		output_dict = param.model_fn(img_tensor)
		for i in range(100):
			if float(output_dict['detection_scores'][0][i]) >= param.score:
				clsidx = int(output_dict['detection_classes'][0][i]) - 1
				if clsidx == 0:
					ngsubfix = '_NG.jpg'
				color = param.colors[clsidx%30]
				tscore = round(100.0*float(output_dict['detection_scores'][0][i]),2)
				bbox = output_dict['detection_boxes'][0][i]
				drawtangle2(bbox,clsidx,cimgx,color,tscore,param)

		# try:
		# 	os.remove(newimgpath)
		# except:
		# 	print('fail to remove file')

		# if '_WAVEGUIDE_' in upperpath and ngsubfix != '_NG.jpg':
		# 	run_model_waveguide11(param)
		# else:
		# 	cv2.imwrite(param.tobepath.replace("test","testout").replace('.jpg',ngsubfix),cimg,[cv2.IMWRITE_JPEG_QUALITY,100])

		cv2.imwrite(param.tobepath.replace("test","testout").replace('.jpg',ngsubfix).replace('.JPG',ngsubfix),cimgx,[cv2.IMWRITE_JPEG_QUALITY,100])
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
	print('python picdie_check.py  --gpuid  '+args.gpuid)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
	gpus = tf.config.list_physical_devices('GPU')
	tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=5*1024)])
	logical_gpus = tf.config.list_logical_devices('GPU')
	print(logical_gpus)

	with tf.device('/device:GPU:0'):
		try:
			paramlist = []
			score = 0.5
			colors = random_colors(30)

			print("loading test data..............")

			data_root = pathlib.Path('./mydata/trainsrcdata/ORION/test1orf1')
			all_image_paths = list(data_root.glob('*'))
			all_image_paths = [str(path) for path in all_image_paths]

			print("loading model files............")

			export_dir = './mydata/trainmodels/ORION/exported_model_OR_8484xxxx'#3f  1f 4199 2f 5863 2p  1p 13
			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8484xxxx1'#3f  1f 4197 2f 5861 2p  1p 14


			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8487xxxx'#3f  1f 4217 2f 5890 2p  1p 14
			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8487xxxx1'#3f  1f 4214 2f 5890 2p  1p 14

			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_850xxxx'#3f  1f 4202 2f 5901  2p  1p 15
			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_850xxxx0'#3f  1f 4195 2f 5897 2p  1p 15




			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8450xxxx'#3f 1532 1f  2p 6 1p
			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8450xxxx0'#3f 1535 1f 4116 2p 6 1p 14

			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8458xxxx'#1f 1542 2f  2p 8


			# export_dir = './mydata/trainmodels/ORION/exported_model_OR_8458xxxx1' 2p 11

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8359xxx1'#1f 1830//2809 2f  1057 1p 4 3f 943 2p 31
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8339xxx0'#1f 2822 2f    1p 2  3f 949  2p 125
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8352xxx0'#1f 1829//2802  2f  1042  1p 2 2p 28

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8339xxx'#1f 2793 2f   1p   3f 943 2p 30
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8364xxx'#1f 1812 2f  1050 2p 


			#mix exported_model_OR_8359xxx1 exported_model_OR_8339xxx0 exported_model_OR_8352xxx0 2892

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8339xxx1'#1f 2783  2f    1p  3f 942
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8359xxx'#1f 1818// 2f  1044  1p 4
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8352xxx'#1f 1841   2f  1045 1p 6
			
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8359xxx0'#1f 1815 2f  1043



			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8318xx'#1f 1555
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8318xx0'#1f 1547

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_OR_8251x'# #h  #r  1m   #1n  1q  1x   2m   1z   1f 2826 2p  1v 1ft 1294 1pt 7 b4-model


			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8225xxxxxxx' # #h  #r  1m   #1n  1q  1x  1347 2m   1z  1389 1f 2815 1p  1v 525 1ft  1pt 10   b4-model
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8225xxxxxxx0' # #h  #r  1m   #1n  1q  1x 1359  2m   1z 1392  1f 2835 1p  1v 536 1ft  1pt 11   b4-model
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8237xxxxxxx' # #h  #r  1m   #1n  1q  1x 1353  2m   1z 1379  1f 2833 1p  1v 528 1ft  1pt  9  b4-model


			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8236xxxxxxx0' #883 #h 67 #r 33 1m   #1n  1q  1x  1354 2m   1z 1395  1f 2830 1p  1v 533 1ft  1pt  7  b4-model			
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8218xxxxxxx' #884 #h 66  #r 28 1m   #1n  1q  1x  1359 2m   1z 1391 1f 2834 1p 7 1v 528 1ft  1pt  7  b4-model
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8232xxxxxxx' # #h 68 #r  1m   #1n  1q  1x 1354  2m   1z 1376  1f 2825 1p 8 1v 535 1ft  1pt 7   b4-model
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8225xxxxxxx0' # #h  #r  1m   #1n  1q  1x 1359  2m   1z 1392  1f 2835 1p  1v 536 1ft  1pt 11   b4-model

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8237xxxxxxx0' # #h  #r  1m   #1n  1q  1x   2m   1z  1376 1f 2823 1p  1v 527 1ft  1pt 10   b4-model
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8236xxxxxxx' #879 #h  #r  1m   #1n  1q  1x 1351  2m   1z  1391 1f 2825 1p  1v 532 1ft  1pt  7  b4-model



			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8218xxxxxxx0' #884 #h  #r  1m   #1n  1q  1x   2m   1z 1382  1f 2832 1p 8  1v 1ft  1pt 8   b4-model
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8232xxxxxxx0' # #h  #r  1m   #1n  1q  1x   2m   1z 1397  1f 2865 1p 18 1v 1ft  1pt 16  b4-model overkill



			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8003xxxxxx'# #h  #r  1m   #1n  1q  1x   2m   1z   1f 2826 2p  1v 1ft 1294 1pt 7 b4-model
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8003xxxxxx0'# #h  #r  1m   #1n  1q  1x   2m   1z   1f  2p  1v 1ft 1282 1pt 9 b4-model

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_7975xxxxxx0'#888 #h 69 #r 30 1m  13 #1n 296 1q 480 1x 1348  2m  8 1z 1375  1f 2827 2p 6 1v 341 1ft 1266 1pt 7 b4-model

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8084xxxxxx'#890 #h 79 #r 30 1m 16  #1n 290 1q 468 1x 1377  2m 12  1z 1390  1f 2859 2p 7 1v 352 1ft 1297 1pt 147 b4-model	 

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8009xxxxxx'#876 #h 66 #r 28 1m 16  #1n 296 1q 478 1x 1349  2m  3  1z  1373 1f 2831  2p 4 1v 341 1pt 22 b4-model	

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8012xxxxxx'#878 #h 64 #r 28 1m 14  #1n 294 1q 470 1x  1349 2m  4  1z 1401  1f  2846 2p 5 1v 341 1pt 41 b4-model	

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_7903xxxxx'#865 #h 66 #r 31 1m 15  #1n 291 1q 468 1x 1344  2m  11  1z  1367 1f  2820  2p 17



			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_7987xxxxxx0'# #h  #r  1m   #1n  1q  1x 1334  2m   1z 1363  1f 2791 2p  1v  b4-model
			#export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_7987xxxxxx'# #h  #r  1m   #1n  1q  1x 1332  2m   1z 1371  1f 2804 2p  1v  b4-model

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_7875xxxxx'#886 #h 66 #r 29 1m 19  #1n 293 1q 470 1x 1358  2m 57 1z 1374  1f 2837  2p 43  b4-model 
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_7865xxxxx'#881 #h 66 #r 29 1m 16  #1n 298 1q 465 1x 1375  2m  25  1z 1400  1f 2881  2p  98 1v 395 (overkill) b4-model	
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_7925xxxxx'#882 #h 68 #r 30 1m 21  #1n 293 1q 468 1x 1364  2m  34  1z 1373 1f   2842  2p 43


			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8030xxxxx'#863 #h 63 #r 31 1m 17  #1n 288 1q 462 1x 1344  2m  6   1z  1351 1f  2802 2p


			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_7901xxxxx'# #h  #r  1m   #1n  1q  1x   2m    1z  1333 1f 2767  b3-model  



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
