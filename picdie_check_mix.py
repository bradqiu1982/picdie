import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import traceback
import sys

# import warnings
# warnings.filterwarnings("ignore")

import logging
logging.getLogger('absl').setLevel('ERROR')

import tensorflow as tf
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=5*1024)])
logical_gpus = tf.config.list_logical_devices('GPU')
print(logical_gpus)

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
	def __init__(self,lock,fn,model_fn,model_fn2,model_fn3,score,colors):
		self.lock = lock
		self.rawpath = fn
		self.model_fn = model_fn
		self.model_fn2 = model_fn2
		self.model_fn3 = model_fn3
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



def CheckOverKill(objbox,objscore,pdboxlist):

	ymin = int(objbox[0])
	xmin = int(objbox[1])
	ymax = int(objbox[2])
	xmax = int(objbox[3])

	xwidth = xmax-xmin
	yheight = ymax-ymin

	xmid = (xmin+xmax)/2
	ymid = (ymin+ymax)/2

	for pdbox in pdboxlist:
		pdymin = int(pdbox[0])
		pdxmin = int(pdbox[1])
		pdymax = int(pdbox[2])
		pdxmax = int(pdbox[3])
		if xmid > pdxmin and xmid < pdxmax and ymid > pdymin and ymid < pdymax:
			return False

	return True

def GetObjectList(output_dict,score):
	wdboxlist = []
	pdboxlist = []
	allngboxlist = []
	allngscorelist = []
	for i in range(100):
		tmpscore = float(output_dict['detection_scores'][0][i])
		clsidx = int(output_dict['detection_classes'][0][i]) - 1
		objbox = output_dict['detection_boxes'][0][i]
		if clsidx == 2:
			if tmpscore >= 0.5:
				wdboxlist.append(objbox)

		if clsidx == 1:
			if tmpscore >= 0.5:
				pdboxlist.append(objbox)

		if clsidx == 0:
			if tmpscore >= score:
				allngscorelist.append(tmpscore)
				allngboxlist.append(objbox)

	return allngscorelist,allngboxlist,pdboxlist,wdboxlist




def run_model(param):
	print("   \r\n")
	print(param.rawpath)

	# newimgpath = param.rawpath
	
	upperpath = param.rawpath.upper()

	gamma = 0.9
	if '_WAVEGUIDE_' in upperpath:
		param.score = 0.4
		gamma = 0.7

	# if '11_METALTRACE' in upperpath or '12_METALTRACE' in upperpath or '13_METALTRACE' in upperpath or '14_METALTRACE' in upperpath or '15_METALTRACE' in upperpath or '16_METALTRACE' in upperpath  or '17_METALTRACE' in upperpath:
	# 	gamma = 1.1

	# if '_MODULATOR_' in upperpath:
	# 	if '29_MODULATOR_' not in upperpath and '30_MODULATOR_' not in upperpath and '31_MODULATOR_' not in upperpath and '32_MODULATOR_' not in upperpath and '33_MODULATOR_' not in upperpath and '34_MODULATOR_' not in upperpath and '35_MODULATOR_' not in upperpath:
	# 		gamma = 1.0

	# if '74_MODULATORPADS_' in upperpath or '75_MODULATORPADS_' in upperpath or '76_MODULATORPADS_' in upperpath or '77_MODULATORPADS_' in upperpath or '78_MODULATORPADS_' in upperpath or '79_MODULATORPADS_' in upperpath or '80_MODULATORPADS_' in upperpath:
	# 	gamma = 1.0

	newimgpath = param.rawpath.replace(".jpg","_0.jpg")
	cimg = cv2.imread(param.rawpath,cv2.IMREAD_COLOR)
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
	

	output_dict1 = param.model_fn(img_tensor)
	output_dict2 = param.model_fn2(img_tensor)
	output_dict3 = param.model_fn3(img_tensor)

	allscorelist1,allngboxlist1,pdboxlist1,wdboxlist1 = GetObjectList(output_dict1,param.score)
	allscorelist2,allngboxlist2,pdboxlist2,wdboxlist2 = GetObjectList(output_dict2,param.score)
	allscorelist3,allngboxlist3,pdboxlist3,wdboxlist3 = GetObjectList(output_dict3,param.score)
	output_dict = {}

	max1 = 0.0
	max2 = 0.0
	max3 = 0.0
	mdrate = 0

	if len(allscorelist1) > 0:
		max1 = max(allscorelist1)
		mdrate = mdrate + 1
	if len(allscorelist2) > 0:
		max2 = max(allscorelist2)
		mdrate = mdrate + 1
	if len(allscorelist3) > 0:
		max3 = max(allscorelist3)
		mdrate = mdrate + 1


	if max1 != 0 and max1 >= max2 and max1 >= max3:
		output_dict = output_dict1
	elif max2 != 0 and max2 >= max1 and max2 >= max3:
		output_dict = output_dict2
	elif max3 != 0 and max3 >= max1 and max3 >= max2:
		output_dict = output_dict3

	ngsubfix = '.jpg'
	cimg = cv2.imread(newimgpath,cv2.IMREAD_COLOR)
	if 'detection_scores' in output_dict:
		for i in range(100):
			if float(output_dict['detection_scores'][0][i]) >= param.score:
				clsidx = int(output_dict['detection_classes'][0][i]) - 1
				if clsidx == 0:
					ngsubfix = '_NG.jpg'
				color = param.colors[clsidx%30]
				tscore = round(100.0*float(output_dict['detection_scores'][0][i]),2)
				bbox = output_dict['detection_boxes'][0][i]
				drawtangle2(bbox,clsidx,cimg,color,tscore)

	try:
		os.remove(newimgpath)
	except:
		print('fail to remove file')

	# if '_WAVEGUIDE_' in upperpath and ngsubfix != '_NG.jpg':
	# 	run_model_waveguide11(param)
	# else:
	# 	ngsubfix = '_'+str(mdrate)+ngsubfix
	# 	cv2.imwrite(param.rawpath.replace("test","testout").replace('.jpg',ngsubfix),cimg,[cv2.IMWRITE_JPEG_QUALITY,100])

	ngsubfix = '_'+str(mdrate)+ngsubfix
	cv2.imwrite(param.rawpath.replace("test","testout").replace('.jpg',ngsubfix),cimg,[cv2.IMWRITE_JPEG_QUALITY,100])

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
	with tf.device('/device:GPU:0'):
		try:
			paramlist = []
			score = 0.5
			colors = random_colors(30)

			print("loading test data..............")

			data_root = pathlib.Path('./mydata/trainsrcdata/PICDIE/test1ms')
			all_image_paths = list(data_root.glob('*'))
			all_image_paths = [str(path) for path in all_image_paths]

			print("loading model files............")


			#exported_model_B91_8030xxxxx#exported_model_B91_7865xxxxx#exported_model_B91_7925xxxxx 1421
			#exported_model_B91_8030xxxxx#exported_model_B91_7865xxxxx#exported_model_B91_7925xxxxx 1424

			export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8218xxxxxxx'
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_7903xxxxx'
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_8030xxxxx'

			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_sys01_788'

			#export_dir = './mydata/trainmodels/PICDIE/exported_model_B91_sys01_781xxx'
			# ****export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_sys01_795'
			# export_dir = './mydata/trainmodels/PICDIE/exported_model_B81_sys01_788'
			# *****export_dir = './mydata/trainmodels/PICDIE/exported_model_B51_sys01_786'
			# *****export_dir = './mydata/trainmodels/PICDIE/exported_model_B51_787'
			imported = tf.saved_model.load(export_dir)
			model_fn = imported.signatures['serving_default']

			export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B91_8232xxxxxxx'
			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B91_7865xxxxx'

			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B91_sys01_786x'

			#export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B91_784' #overkill
			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B81_789'
			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B81_802'
			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B71_808'#14
			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B71_805'#15
			# *****export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B52_765'
			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B71_793'
			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_B61_77x'
			imported2 = tf.saved_model.load(export_dir2)
			model_fn2 = imported2.signatures['serving_default']

			
			export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B91_8236xxxxxxx0'
			# export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B91_7925xxxxx'

			# export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B91_sys01_781xxx'

			# export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B91_sys01_778xxx'
			# export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B91_sys01_785xx'
			# export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B91_791'
			# export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B91_sys01_786'
			# export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B81_786'
			# export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B81_sys01_789x'
			# ***export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B71_808'
			# *****export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B51_sys01_786'
			# export_dir3 = './mydata/trainmodels/PICDIE/exported_model_B52_765'
			imported3 = tf.saved_model.load(export_dir3)
			model_fn3 = imported3.signatures['serving_default']



			for fn in all_image_paths:
				if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
					param = ProcParam(wblock,fn,model_fn,model_fn2,model_fn3,score,colors)
					paramlist.append(param)

			print("get test data count: "+str(len(paramlist)))

			for param in paramlist:
				run_model(param)

		except:
			print('run into exception0.......')
			exception_message = sys.exc_info()[1]
			print(str(exception_message))
			traceback.print_exc()

if __name__ == "__main__":
	MainLoop()
