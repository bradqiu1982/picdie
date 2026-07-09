
import os
import sys
import tensorflow as tf
import io
import numpy as np
import pathlib
import cv2


label_names = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
HIGH = 64
WIDTH = 64
input_image_size = (HIGH, WIDTH)


model = tf.saved_model.load('D:/PlanningForCast/condaenv/VISION/models-master/mydata/exported_model_bdcimgcls')
model_fn = model.signatures['serving_default']


data_root = pathlib.Path('D:/PlanningForCast/condaenv/VISION/models-master/mydata/OOD-DATATOOL/BDC_TrainData/test')
all_image_paths = list(data_root.glob('*'))
#all_image_paths = [str(path) for path in all_image_paths]
for fn in all_image_paths:
	filename = fn.name
	fnpath = str(fn)
	if ('.JPG' in fnpath.upper() or '.JPEG' in fnpath.upper() or '.PNG' in fnpath.upper() or '.BMP' in fnpath.upper()):
		print(fnpath)

		img = tf.io.read_file(fnpath)
		img_tensor = tf.io.decode_image(img, channels=3)
		img_tensor = tf.image.resize(img_tensor,input_image_size)
		img_tensor = tf.expand_dims(img_tensor, axis=0)
		img_tensor = tf.cast(img_tensor, dtype = tf.uint8)

		output_dict = model_fn(img_tensor)
		index = (tf.argmax(output_dict['logits'], axis=1)[0]).numpy()
		clsstr = label_names[index]

		cimg = cv2.imread(fnpath,cv2.IMREAD_COLOR)
		cimg = cv2.resize(cimg,(WIDTH,HIGH))
		localfn = './mydata/OOD-DATATOOL/BDC_TrainData/test-classed/'+clsstr+'/'+filename
		print(localfn)
		cv2.imwrite(localfn,cimg)

