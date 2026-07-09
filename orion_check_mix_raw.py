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


from absl import app
from absl import flags
from absl import logging



import dataclasses
from typing import Optional, List, Sequence, Union

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

	cv2.putText(cimg,ref_dict[int(cls)]+','+str(xwidth)+','+str(yheight)+','+str(tscore),(xmark,ymark),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
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


def GetObjectList(output_dict,score,allngboxlist,allpdboxlist80,allpdboxlist90,allwdboxlist80,allwdboxlist90,allngscorelist):
	tempngscorelist = []
	for i in range(100):
		tmpscore = float(output_dict['detection_scores'][0][i])
		clsidx = int(output_dict['detection_classes'][0][i]) - 1
		objbox = output_dict['detection_boxes'][0][i]

		if clsidx == 2:
			if tmpscore >= 0.9:
				allwdboxlist90.append(objbox)
			elif tmpscore >= 0.75:
				allwdboxlist80.append(objbox)

		if clsidx == 1:
			if tmpscore >= 0.9:
				allpdboxlist90.append(objbox)
			elif tmpscore >= 0.75:
				allpdboxlist80.append(objbox)

		if clsidx == 0:
			if tmpscore >= score:
				allngscorelist.append(tmpscore)
				tempngscorelist.append(tmpscore)
				allngboxlist.append(objbox)
	return tempngscorelist

def checkboxdist(boxlist,distthold1=54,distthold2=160):
	for box1 in boxlist:
		for box2 in boxlist:
			ymin1 = int(box1[0])
			xmin1 = int(box1[1])
			ymax1 = int(box1[2])
			xmax1 = int(box1[3])

			ymin2 = int(box2[0])
			xmin2 = int(box2[1])
			ymax2 = int(box2[2])
			xmax2 = int(box2[3])

			p = [int((xmin1+xmax1)/2),int((ymin1+ymax1)/2)]
			q = [int((xmin2+xmax2)/2),int((ymin2+ymax2)/2)]
			midptdist = math.dist(p,q)
			if midptdist < 10:
				continue

			p = [xmin1,ymin1]
			q = [xmax2,ymax2]
			dist1 = math.dist(p,q)

			p = [xmax1,ymin1]
			q = [xmin2,ymax2]
			dist2 = math.dist(p,q)

			print("box dist1..."+str(dist1)+"... dist2..."+str(dist2))

			if (dist1 > distthold1 and dist1 < distthold2) or (dist2 > distthold1 and dist2 < distthold2) :
				return True
	return False


def rawWH(box):
	ymin = int(box[0]*2048/1280)
	xmin = int(box[1]*2448/1280)
	ymax = int(box[2]*2048/1280)
	xmax = int(box[3]*2448/1280)
	xwidth = int(float(xmax-xmin)/120.0*40.0)
	yheight = int(float(ymax-ymin)/120.0*40.0)
	return xwidth,yheight

def SameBox(boxlist,distthold1=20):
	if len(boxlist) == 1:
		return True

	for box1 in boxlist:
		for box2 in boxlist:
			ymin1 = int(box1[0])
			xmin1 = int(box1[1])
			ymax1 = int(box1[2])
			xmax1 = int(box1[3])

			ymin2 = int(box2[0])
			xmin2 = int(box2[1])
			ymax2 = int(box2[2])
			xmax2 = int(box2[3])

			p = [int((xmin1+xmax1)/2),int((ymin1+ymax1)/2)]
			q = [int((xmin2+xmax2)/2),int((ymin2+ymax2)/2)]
			dist1 = math.dist(p,q)

			print('same box dist1: '+str(dist1)+'..........')

			if dist1 > distthold1:
				return False
	return True

def CheckDefectSize(boxlist,tempngboxlist,defectsize=36):
	if len(tempngboxlist) > 0 and SameBox(tempngboxlist):
		w = -1
		h = -1
		for box in tempngboxlist:
			w1,h1 = rawWH(box)
			if w1 > w:
				w = w1
			if h1 > h:
				h = h1

		# print('tw: '+ str(w) + ' h: '+str(h)+'..................')
		if w < defectsize and h < defectsize: #and (w*w+h*h) < defectsize*defectsize:
			return True
	return False


def AROUNDWAVEGUIDE(imgpath,allngboxlist,output_dict,mdrate,allngscorelist):
	boxlist = []
	tempngboxlist = []

	idx = 0
	for objbox in allngboxlist:
		score = allngscorelist[idx]
		idx = idx + 1

		ymin = int(objbox[0])
		xmin = int(objbox[1])
		ymax = int(objbox[2])
		xmax = int(objbox[3])
		xmid = (xmin+xmax)/2
		ymid = (ymin+ymax)/2
		xwidth,yheight = rawWH(objbox)


		if '4_AROUNDWAVEGUIDE' in imgpath:
			if  xmid > 280 and xmid < 640 and yheight <= 13:
				continue

		if '1_AROUNDWAVEGUIDE' in imgpath:

			if xmid > 860:
				if ymid > 200 and ymid < 460:
					if xwidth > 20 and yheight > 20:
						boxlist.append(objbox)
						continue
					if xwidth > 30 or yheight > 30:
						boxlist.append(objbox)
						continue

				if ymid > 460:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

			if xmid > 350 and xmid < 860:
				if ymid < 750:
					if ymid > 300 and ymid < 400:
						if xwidth > 55 or yheight > 55:
							boxlist.append(objbox)
							continue

					if xwidth >= 50 and yheight >= 50:
						boxlist.append(objbox)
						continue
				else:
					if xwidth >= 130 or yheight >= 130:
						boxlist.append(objbox)
						continue

		elif '2_AROUNDWAVEGUIDE' in imgpath or '3_AROUNDWAVEGUIDE' in imgpath or '4_AROUNDWAVEGUIDE' in imgpath:

			if ymid > 250 and ymid < 400:
				if xwidth >= 20 or yheight >= 20:
					boxlist.append(objbox)
					continue

			if ymid > 200 and ymid < 460:
				if xwidth > 20 and yheight > 20:
					boxlist.append(objbox)
					continue
				if xwidth > 30 or yheight > 30:
						boxlist.append(objbox)
						continue

			if ymid > 460 and ymid < 750 :
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 20 or yheight >= 20:
					tempngboxlist.append(objbox)
					continue

			if  '4_AROUNDWAVEGUIDE' in imgpath and ymid > 1000:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 20 or yheight >= 20:
					tempngboxlist.append(objbox)
					continue

		elif '7_AROUNDWAVEGUIDE' in imgpath:
			if xmid >570 and xmid < 960:
				if ymid > 250 and ymid < 400:
					if xwidth >= 20 or yheight >= 20:
						boxlist.append(objbox)
						continue

				if ymid > 200 and ymid < 460:
					if xwidth > 20 and yheight > 20:
						boxlist.append(objbox)
						continue
					if xwidth > 30 or yheight > 30:
							boxlist.append(objbox)
							continue

			if xmid > 800 and ymid > 250 and ymid < 410:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue

			if xmid < 290:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue

				if xwidth >= 20 or yheight >= 20:
					tempngboxlist.append(objbox)
					continue

			if ymid > 1040:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 20 or yheight >= 20:
					tempngboxlist.append(objbox)
					continue

		elif '8_AROUNDWAVEGUIDE' in imgpath:

			if xmid > 1240 and xwidth <= 10:
				continue


			if xmid >300 and xmid < 640:
				if ymid > 250 and ymid < 400:
					if xwidth >= 20 or yheight >= 20:
						boxlist.append(objbox)
						continue

				if ymid > 200 and ymid < 460:
					if xwidth > 20 and yheight > 20:
						boxlist.append(objbox)
						continue
					if xwidth > 30 or yheight > 30:
							boxlist.append(objbox)
							continue

			if xmid > 520 and ymid > 250 and ymid < 410:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue

			if  ymid > 1040:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 20 or yheight >= 20:
					tempngboxlist.append(objbox)
					continue
		else:
			if (ymid > 700 and ymid < 900) or ymid > 1000:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 20 or yheight >= 20:
					tempngboxlist.append(objbox)
					continue

	if len(boxlist) > 0:
		return output_dict,mdrate,boxlist 

	if CheckDefectSize(boxlist,tempngboxlist):
		return {},mdrate,[]

	if len(tempngboxlist) > 0:
		if checkboxdist(tempngboxlist):
			return output_dict,mdrate,boxlist
	else:
		return {},mdrate,[]

	return {},mdrate,[]

def METALTRACE(imgpath,allngboxlist,output_dict,mdrate,allngscorelist):
	boxlist = []
	tempngboxlist = []

	idx = 0
	for objbox in allngboxlist:
		score = allngscorelist[idx]
		idx = idx + 1

		ymin = int(objbox[0])
		xmin = int(objbox[1])
		ymax = int(objbox[2])
		xmax = int(objbox[3])
		xmid = (xmin+xmax)/2
		ymid = (ymin+ymax)/2
		xwidth,yheight = rawWH(objbox)

		if (xmid < 30 or xmid > 1250) and xwidth <= 10:
			continue

		if '11_METALTRACE' in imgpath:
			#in upper
			if ymid < 210:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 18 or yheight >= 18:
					tempngboxlist.append(objbox)
					continue
			#right
			if xmid > 1000:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 18 or yheight >= 18:
					tempngboxlist.append(objbox)
					continue

			#in middle
			if ymid > 420 and ymid < 865:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 18 or yheight >= 18:
					tempngboxlist.append(objbox)
					continue
			#bottom
			if ymid > 1200:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 18 or yheight >= 18:
					tempngboxlist.append(objbox)
					continue

		elif '18_METALTRACE' in imgpath:
			if xmid > 350 and xmid < 860:
				if yheight > 210:
					continue

				if xwidth >= 130 or yheight >= 130:
					boxlist.append(objbox)
					continue
			if xmid > 860:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 18 or yheight >= 18:
					tempngboxlist.append(objbox)
					continue
		else:

			if '17_METALTRACE' in imgpath:
				if ymid < 80:
					continue

			#in upper
			if ymid < 220:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 18 or yheight >= 18:
					tempngboxlist.append(objbox)
					continue
			#middle
			if ymid > 420 and ymid < 865:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 18 or yheight >= 18:
					tempngboxlist.append(objbox)
					continue
			#bottom
			if ymid > 1200:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 18 or yheight >= 18:
					tempngboxlist.append(objbox)
					continue

	print("boxlist len is "+str(len(boxlist)))

	if len(boxlist) > 0:
		return output_dict,mdrate,boxlist 

	print("tempngboxlist len is "+str(len(tempngboxlist)))

	if CheckDefectSize(boxlist,tempngboxlist):
		return {},mdrate,[]

	print("check box dist.........")

	if len(tempngboxlist) > 0:
		if checkboxdist(tempngboxlist):
			return output_dict,mdrate,boxlist
	else:
		return {},mdrate,[]

	return {},mdrate,[]

def MODULATORHEATER(imgpath,allngboxlist,output_dict,mdrate,allngscorelist,allwdboxlist):

	boxlist = []
	tempngboxlist = []

	sumcnt = 0
	sumymin = 0
	sumymax = 0
	avgwdymin = -1
	avgwdymax = -1
	leftwd = 1280
	rightwd = 0

	for wdbox in allwdboxlist:
		sumymin = sumymin+int(wdbox[0])
		sumymax = sumymax+int(wdbox[2])
		sumcnt = sumcnt+1
		if int(wdbox[1]) < leftwd:
			leftwd =  int(wdbox[1])
		if int(wdbox[3]) > rightwd:
			rightwd = int(wdbox[3])

	if sumcnt != 0:
		avgwdymin = int(sumymin/sumcnt)
		avgwdymax = int(sumymax/sumcnt)

		boxlist = []
		tempngboxlist = []
		idx = 0
		for objbox in allngboxlist:
			score = allngscorelist[idx]
			idx = idx + 1

			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2
			xwidth,yheight = rawWH(objbox)

			if ymid > avgwdymax+450 and ymid < avgwdymax+520 and yheight <= 21:
				continue
			if ymid <= 20 and yheight <=20:
				continue

			if '19_MODULATOR_'  in imgpath:
				if xmid > 1250 and ymid < 500 and xwidth <= 10:
					continue

				#in wd
				if xmid > leftwd-10 and xmid < leftwd+100 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						boxlist.append(objbox)
						continue
				#right
				if xmid > leftwd-30:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue
				#on edge
				if xmid > leftwd-450 and xmid < leftwd-550:
					if xwidth > 39 and yheight > 39:
						boxlist.append(objbox)
						continue

			elif '20_MODULATOR_'  in imgpath:
				if leftwd > 600:
					leftwd = rightwd - 875
				else:
					rightwd = leftwd + 875

				#in wd
				if xmid > leftwd-10 and xmid < leftwd+100 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				if xmid > rightwd - 100 and xmid < rightwd+10 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				#left right
				if xmid < leftwd+100 or xmid > rightwd - 100:
					if ymid > avgwdymax+270:
						if xwidth >= 35 or yheight >= 35:
							boxlist.append(objbox)
							continue
						if xwidth > 15 or yheight > 15:
							tempngboxlist.append(objbox)
							continue
					else:
						if xwidth > 38 or yheight > 38:
							boxlist.append(objbox)
							continue
						if xwidth >= 20 or yheight >= 20:
							tempngboxlist.append(objbox)
							continue

				#in upper
				if ymid < avgwdymin-300:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#in middle
				if ymid > avgwdymax+50 and ymid < avgwdymax+270:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

			elif '21_MODULATOR_'  in imgpath:
				if leftwd < 550:
					rightwd = leftwd+875
				elif leftwd > 1050:
					rightwd = leftwd-265
					leftwd = rightwd-875
				else:
					rightwd = leftwd+90
					leftwd = rightwd-875

				if xmid <= 20 and xwidth <=15:
					continue

				#in wd
				if xmid > leftwd-10 and xmid < leftwd+100 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				if xmid > rightwd - 100 and xmid < rightwd+10 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				if xmid > rightwd +255  and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				#in upper
				if ymin < avgwdymin-275:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#in middle
				if ymid > avgwdymax+50 and ymid < avgwdymax+270:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#left and right
				if xmid < leftwd+100 or xmid > rightwd - 100:
					if ymid > avgwdymax+270:
						if xwidth >= 35 or yheight >= 35:
							boxlist.append(objbox)
							continue
						if xwidth > 15 or yheight > 15:
							tempngboxlist.append(objbox)
							continue
					else:
						if xwidth > 38 or yheight > 38:
							boxlist.append(objbox)
							continue
						if xwidth >= 20 or yheight >= 20:
							tempngboxlist.append(objbox)
							continue

			elif '22_MODULATOR_'  in imgpath:
				if rightwd < 550:
					rightwd=rightwd+785
					leftwd = rightwd-875
				elif rightwd > 1050:
					rightwd = rightwd-350
					leftwd = rightwd-875
				else:
					leftwd = rightwd-875

				#in wd
				if xmid < leftwd+100 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				if xmid > rightwd - 100 and xmid < rightwd+10 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				if xmid > rightwd +255 and xmid < rightwd +355  and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				#in upper
				if ymin < avgwdymin-240:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#in middle
				if ymid > avgwdymax+50 and ymid < avgwdymax+270:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#left and right
				if xmid < leftwd+100 or (xmid > rightwd - 100 and xmid < rightwd + 365):
					if ymid > avgwdymax+270:
						if xwidth >= 35 or yheight >= 35:
							boxlist.append(objbox)
							continue
						if xwidth > 15 or yheight > 15:
							tempngboxlist.append(objbox)
							continue
					else:
						if xwidth > 38 or yheight > 38:
							boxlist.append(objbox)
							continue
						if xwidth >= 20 or yheight >= 20:
							tempngboxlist.append(objbox)
							continue

			elif '26_MODULATOR_'  in imgpath:
				if leftwd > 630:
					leftwd = rightwd - 430
				else:
					rightwd = leftwd + 430

				if xmid > rightwd + 285 and xmid < rightwd + 345 and xwidth <= 18:
					continue

				if xmid > 1240 and xwidth <= 10:
					continue

				#in wd
				if xmid > leftwd-10 and xmid < leftwd+100 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				if xmid > rightwd - 100 and xmid < rightwd+10 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				#in upper
				if ymin < avgwdymin-145:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#in middle
				if ymid > avgwdymax+50 and ymid < avgwdymax+270:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#right
				if xmid > leftwd-10:
					if ymid > avgwdymax+270 and xmid > leftwd and xmid < leftwd+360:
						if xwidth >= 35 or yheight >= 35:
							boxlist.append(objbox)
							continue
						if xwidth > 15 or yheight > 15:
							tempngboxlist.append(objbox)
							continue
					else:
						if xwidth > 38 or yheight > 38:
							boxlist.append(objbox)
							continue
						if xwidth >= 20 or yheight >= 20:
							tempngboxlist.append(objbox)
							continue
			else:
				if rightwd-leftwd > 200:
					tmpx = 0
				else:
					if rightwd > 860:
						leftwd = rightwd - 430
					else:
						rightwd = leftwd+430

				#in wd
				if xmid > leftwd-10 and xmid < leftwd+100 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				if xmid > rightwd - 100 and xmid < rightwd+10 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
					if xwidth > 15 and yheight > 15:
						boxlist.append(objbox)
						continue
					if xwidth > 20 or yheight > 20:
						boxlist.append(objbox)
						continue

				#in upper
				if ymin < avgwdymin-200:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#in middle
				if ymid > avgwdymax+50 and ymid < avgwdymax+270:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#in middle
				if xmid > leftwd-10 and xmid< rightwd+10:
					if ymid > avgwdymax+270:
						if xwidth >= 35 or yheight >= 35:
							boxlist.append(objbox)
							continue
						if xwidth > 15 or yheight > 15:
							tempngboxlist.append(objbox)
							continue
					else:
						if xwidth > 38 or yheight > 38:
							boxlist.append(objbox)
							continue
						if xwidth >= 20 or yheight >= 20:
							tempngboxlist.append(objbox)
							continue

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist 

		if CheckDefectSize(boxlist,tempngboxlist):
			return {},mdrate,[]

		if len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	else:

		boxlist = []
		tempngboxlist = []
		idx = 0
		for objbox in allngboxlist:
			score = allngscorelist[idx]
			idx = idx + 1

			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2
			xwidth,yheight = rawWH(objbox)

			if xwidth > 35 or yheight > 35:
				boxlist.append(objbox)
				continue
			if xwidth > 15 or yheight > 15:
				tempngboxlist.append(objbox)
				continue

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist 

		if CheckDefectSize(boxlist,tempngboxlist):
			return {},mdrate,[]

		if len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	return {},mdrate,[]

def MODULATOR(imgpath,allngboxlist,output_dict,mdrate,allngscorelist):

	boxlist = []
	tempngboxlist = []

	idx = 0
	for objbox in allngboxlist:
		score = allngscorelist[idx]
		idx = idx + 1

		ymin = int(objbox[0])
		xmin = int(objbox[1])
		ymax = int(objbox[2])
		xmax = int(objbox[3])
		xmid = (xmin+xmax)/2
		ymid = (ymin+ymax)/2
		xwidth,yheight = rawWH(objbox)

		if (ymid <=20 or ymid >= 1260) and yheight <= 12:
			continue
		
		if '36_MODULATOR' in imgpath or '37_MODULATOR' in imgpath or '54_MODULATOR' in imgpath or '55_MODULATOR' in imgpath:
			if xmid > 1050:
				if xwidth >= 23 or yheight >= 23 or ((xwidth*xwidth+yheight*yheight) >= 28*28):
					boxlist.append(objbox)
					continue

				if xwidth > 15 or yheight > 15:
					tempngboxlist.append(objbox)
					continue
			elif xmid > 400:
				if xwidth > 50 or yheight > 50:
					boxlist.append(objbox)
					continue

		elif '29_MODULATOR' in imgpath or '44_MODULATOR' in imgpath or '47_MODULATOR' in imgpath or '62_MODULATOR' in imgpath:
			if xmid > 1250 and xwidth <= 13:
				continue

			leftbond0 = 420
			rightbond0 = leftbond0+420

			if xmid > leftbond0+100 and xmid < rightbond0-100:
				if xwidth >= 23 or yheight >= 23 or ((xwidth*xwidth+yheight*yheight) >= 28*28):
					boxlist.append(objbox)
					continue

			if xmid > leftbond0 and xmid < rightbond0:
				if xwidth >= 25 or yheight >= 25 or ((xwidth*xwidth+yheight*yheight) >= 32*32):
					boxlist.append(objbox)
					continue
				if xwidth > 15 or yheight > 15:
					tempngboxlist.append(objbox)
					continue
			if xmid > 820:
				if xwidth > 38 or yheight > 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 20 or yheight >= 20:
					tempngboxlist.append(objbox)
					continue
		else:
			leftbond0 = 0
			rightbond0 = 1280
			leftbond1 = -150
			rightbond1 = -150

			if '30_MODULATOR' in imgpath or '48_MODULATOR' in imgpath or '61_MODULATOR' in imgpath :
				leftbond0 = 515
			elif '31_MODULATOR' in imgpath or '42_MODULATOR' in imgpath or '49_MODULATOR' in imgpath or '60_MODULATOR' in imgpath:
				leftbond0 = 615
			elif '32_MODULATOR'  in imgpath or '41_MODULATOR'  in imgpath or  '50_MODULATOR' in imgpath or  '59_MODULATOR' in imgpath:
				leftbond0 = 705
			elif '33_MODULATOR' in imgpath or  '57_MODULATOR' in imgpath:
				leftbond0 = 775
			elif '34_MODULATOR'  in imgpath or '39_MODULATOR' in imgpath or '52_MODULATOR'  in imgpath:
				leftbond0 = 890
				leftbond1 = 0
				rightbond1 = 155

				if xmid > 1240 and xwidth < 10:
					continue

			elif '35_MODULATOR'  in imgpath or '38_MODULATOR' in imgpath or '53_MODULATOR' in imgpath:
				leftbond0 = 975
				leftbond1 = 0
				rightbond1 = 235
			elif '40_MODULATOR' in imgpath or '51_MODULATOR' in imgpath or '58_MODULATOR' in imgpath:
				leftbond0 = 795
			elif '43_MODULATOR' in imgpath:
				leftbond0 = 495
			elif '56_MODULATOR' in imgpath:
				leftbond0 = 945
				leftbond1 = 0
				rightbond1 = 205
			
			rightbond0 = leftbond0+420

			if (xmid > leftbond0+100 and xmid < rightbond0-100) or (xmid < rightbond1-100):
				if xwidth >= 23 or yheight >= 23 or ((xwidth*xwidth+yheight*yheight) >= 28*28):
					boxlist.append(objbox)
					continue

			if (xmid > leftbond0 and xmid < rightbond0) or (xmid > leftbond1 and xmid < rightbond1):
				if xwidth >= 25 or yheight >= 25 or ((xwidth*xwidth+yheight*yheight) >= 32*32):
					boxlist.append(objbox)
					continue
				if xwidth > 15 or yheight > 15:
					tempngboxlist.append(objbox)
					continue

	if len(boxlist) > 0:
		return output_dict,mdrate,boxlist 

	if CheckDefectSize(boxlist,tempngboxlist):
		return {},mdrate,[]

	if len(tempngboxlist) > 0:
		if checkboxdist(tempngboxlist):
			return output_dict,mdrate,boxlist
	else:
		return {},mdrate,[]

	return {},mdrate,[]


def MODULATORPADS(imgpath,allngboxlist,output_dict,mdrate,allngscorelist,allpdboxlist):

	sumcnt = 0
	sumymin = 0
	sumymax = 0
	avgpdymin = -1
	avgpdymax = -1

	for pdbox in allpdboxlist:
		sumymin = sumymin+int(pdbox[0])
		sumymax = sumymax+int(pdbox[2])
		sumcnt = sumcnt+1

	if sumcnt != 0:
		avgpdymin = int(sumymin/sumcnt)
		avgpdymax = int(sumymax/sumcnt)

		boxlist = []
		tempngboxlist = []

		idx = 0
		for objbox in allngboxlist:
			score = allngscorelist[idx]
			idx = idx + 1

			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2
			xwidth,yheight = rawWH(objbox)


			if '70_MODULATORPADS'  in imgpath:
				if xmid > 1230 and ymid < 500:
					continue
				if xmid < 20 and xwidth <= 20 and ymid > avgpdymin-200 and ymid < avgpdymin -20:
					continue

			if '65_MODULATORPADS'  in imgpath:
				if xmid > 1230 and xwidth <= 15:
					continue

			leftbond0 = 0
			rightbond0 = 1280
			leftbond1 = -150
			rightbond1 = -150

			if '65_MODULATORPADS'  in imgpath:
				leftbond0 = 430
			elif '66_MODULATORPADS'  in imgpath:
				leftbond0 = 520
			elif '67_MODULATORPADS'  in imgpath:
				leftbond0 = 620
			elif '68_MODULATORPADS'  in imgpath:
				leftbond0 = 685
			elif '69_MODULATORPADS'  in imgpath:
				leftbond0 = 800
			elif '70_MODULATORPADS'  in imgpath:
				leftbond0 = 915
				leftbond1 = 0
				rightbond1 = 190
			elif '71_MODULATORPADS'  in imgpath:
				leftbond0 = 950
				leftbond1 = 0
				rightbond1 = 220
			elif '72_MODULATORPADS'  in imgpath:
				leftbond0 = 1025
			rightbond0 = leftbond0+420

			#in upper
			if ymid < avgpdymin - 225:

				if (xmid > leftbond0+100 and xmid < rightbond0-100) or (xmid < rightbond1-100):
					if xwidth >= 23 or yheight >= 23 or ((xwidth*xwidth+yheight*yheight) >= 28*28):
						boxlist.append(objbox)
						continue

				if (xmid > leftbond0 and xmid < rightbond0) or (xmid > leftbond1 and xmid < rightbond1):
					if xwidth >= 25 or yheight >= 25 or ((xwidth*xwidth+yheight*yheight) >= 32*32):
						boxlist.append(objbox)
						continue
					if xwidth > 15 or yheight > 15:
						tempngboxlist.append(objbox)
						continue

			#in middle
			if ymid >= avgpdymin - 225 and ymid <= avgpdymax:
				if xwidth >= 38 or yheight >= 38:
					boxlist.append(objbox)
					continue
				if xwidth >= 20 or yheight >= 20:
					tempngboxlist.append(objbox)
					continue


			# if '72_MODULATORPADS' in imgpath:
			# 	if ymid > avgpdymax+40 and ymid <  avgpdymax-40 and xmid > 450 and xmid < 550:
			# 		if xwidth >= 25 or yheight >= 25:
			# 			boxlist.append(objbox)
			# 			continue

			heighthold = 48
			if '72_MODULATORPADS' in imgpath and xmid < 620:
				heighthold = 63
			#in bottom
			if ymid > avgpdymax and ymid < avgpdymax+200:
				if xwidth > 39 and yheight > heighthold:
					boxlist.append(objbox)
					continue
				# if xwidth > 15 or yheight > 15:
				# 	tempngboxlist.append(objbox)
				# 	continue

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist 

		if CheckDefectSize(boxlist,tempngboxlist):
			return {},mdrate,[]

		if len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

		return {},mdrate,[]
	else:
		boxlist = []
		tempngboxlist = []

		idx = 0
		for objbox in allngboxlist:
			score = allngscorelist[idx]
			idx = idx + 1

			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2
			xwidth,yheight = rawWH(objbox)

			if ymid < 490:
				if xwidth > 23 and yheight > 23:
					boxlist.append(objbox)
					continue
				if xwidth > 32 or yheight > 32:
					boxlist.append(objbox)
					continue
				if xwidth > 15 or yheight > 15:
					tempngboxlist.append(objbox)
					continue

			if ymid >= 490 and ymid <= 860:
				if xwidth >= 35 or yheight >= 35:
					boxlist.append(objbox)
					continue
				if xwidth > 15 or yheight > 15:
					tempngboxlist.append(objbox)
					continue

			if ymid > 860 and ymid <= 1060:
				if xwidth > 39 and yheight > 55:
					boxlist.append(objbox)
					continue
				# if xwidth > 15 or yheight > 15:
				# 	tempngboxlist.append(objbox)
				# 	continue

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist

		if CheckDefectSize(boxlist,tempngboxlist):
			return {},mdrate,[]

		if len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

		return {},mdrate,[]

def CONTROLPADS(imgpath,allngboxlist,output_dict,mdrate,allngscorelist,allpdboxlist):

	boxlist = []
	tempngboxlist = []

	sumcnt = 0
	sumleft = 0
	sumright = 0
	sumarea = 0

	padhead = 1280
	padbtm = 0

	for pdbox in allpdboxlist:
		pdymin = int(pdbox[0])
		pdxmin = int(pdbox[1])
		pdymax = int(pdbox[2])
		pdxmax = int(pdbox[3])
		pdyheight = pdymax-pdymin
		pdxwidth = pdxmax-pdxmin

		if pdxmax > 550  and pdyheight > 150:
			sumleft = sumleft + pdxmin
			sumright = sumright + pdxmax
			sumarea = sumarea+pdxwidth*pdyheight
			sumcnt = sumcnt + 1
			if pdymin < padhead:
				padhead = pdymin
			if pdymax > padbtm:
				padbtm = pdymax

	if sumcnt != 0:
		avgleft = int(sumleft/sumcnt)
		avgright = int(sumright/sumcnt)
		minarea = int(float(sumarea/sumcnt)*0.2)
		
		boxlist = []
		tempngboxlist = []
		
		if '9_CONTROLPADS'  in imgpath:
			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2
				xwidth,yheight = rawWH(objbox)

				#in upper
				if ymid > padhead-660 and ymid < padhead-450:
					if xmid < 330:
						if xwidth > 20 or yheight > 20:
							boxlist.append(objbox)
							continue
						if xwidth > 15 or yheight > 15:
							tempngboxlist.append(objbox)
							continue
					else:
						if xwidth >= 35 or yheight >= 35:
							boxlist.append(objbox)
							continue

				# if ymid > padhead-610 and ymid < padhead - 430 and xmid > avgright-30  and xmid < avgright+70:
				# 	if xwidth >= 20 or yheight >= 20:
				# 		boxlist.append(objbox)
				# 		continue

				#in edage
				if xmid > avgright and xmid < avgright+100 and ymid > padhead-450:
					if xwidth > 49 and yheight > 49:
						boxlist.append(objbox)
						continue
					# if xwidth > 15 or yheight > 15:
					# 	tempngboxlist.append(objbox)
					# 	continue

				#left bottom
				if ymid > padhead-70 and xmid < avgleft:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#on pad
				if xmid > avgleft-30 and xmid < avgright+30 and ymid > padhead-30 and ymid < padbtm+30:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist 

			if CheckDefectSize(boxlist,tempngboxlist):
				return {},mdrate,[]

			if len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '64_CONTROLPADS'  in imgpath:
			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2
				xwidth,yheight = rawWH(objbox)

				if padbtm > 420:
					padhead = padbtm - 455
				elif padhead < 210:
					padbtm = padhead + 455


				#on bottom
				if ymid > padbtm+280 and ymid < padbtm+430:
					if xmid > avgleft - 30 and xmid < avgright+150:
						if xwidth > 39 and yheight > 63:
							boxlist.append(objbox)
							continue
						# if xwidth > 15 or yheight > 15:
						# 	tempngboxlist.append(objbox)
						# 	continue

					if xmid <= avgleft - 30:
						if xwidth > 39 and yheight > 50:
							boxlist.append(objbox)
							continue

				# if ymid > padbtm+240 and ymid < padbtm+370 and xmid > avgright-40 and xmid < avgright+70:
				# 	if xwidth > 20 or yheight > 20:
				# 		boxlist.append(objbox)
				# 		continue

				#in pad
				if xmid > avgleft-30 and xmid < avgright+30 and ymid > padhead-30 and ymid < padbtm+30:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#left bottom
				if xmid < avgleft-350 and ymid > padbtm+150 and ymid < padbtm+430:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

				#on edage
				if xmid > avgright and xmid < avgright+100 and ymid > padbtm+430:
					if xwidth > 49 and yheight > 49:
						boxlist.append(objbox)
						continue
					# if xwidth > 15 or yheight > 15:
					# 	tempngboxlist.append(objbox)
					# 	continue


			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist 

			if CheckDefectSize(boxlist,tempngboxlist):
				return {},mdrate,[]

			if len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		else:
			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2
				xwidth,yheight = rawWH(objbox)

				if xmid < 40 and xwidth < 10:
					continue

				#on edge
				if xmid > avgright and xmid < avgright+100:
					if xwidth > 39 and yheight > 39:
						boxlist.append(objbox)
						continue
					# if xwidth > 15 or yheight > 15:
					# 	tempngboxlist.append(objbox)
					# 	continue

				#on pad
				if xmid > avgleft-30 and xmid < avgright+30:
					if (xwidth > 38 or yheight > 38) and yheight > 12:
						boxlist.append(objbox)
						continue
					if (xwidth >= 20 or yheight >= 20) and yheight > 12:
						tempngboxlist.append(objbox)
						continue

				#left
				if xmid < avgleft-30:
					if xwidth > 38 or yheight > 38:
						boxlist.append(objbox)
						continue
					if xwidth >= 20 or yheight >= 20:
						tempngboxlist.append(objbox)
						continue

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist 

			if CheckDefectSize(boxlist,tempngboxlist):
				return {},mdrate,[]

			if len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]

	else:

		boxlist = []
		tempngboxlist = []

		idx = 0
		for objbox in allngboxlist:
			score = allngscorelist[idx]
			idx = idx + 1

			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2
			xwidth,yheight = rawWH(objbox)

			# if score < 0.6:
			# 	continue

			if xmid < 850:
				if '9_CONTROLPADS'  in imgpath:
					if ymid > 200:
						if ymid > 500:
							if xwidth > 25 or yheight > 25:
								boxlist.append(objbox)
								continue
							if xwidth > 15 or yheight > 15:
								tempngboxlist.append(objbox)
								continue

						if xwidth >= 35 or yheight >= 35:
							boxlist.append(objbox)
							continue
						if xwidth > 15 or yheight > 15:
							tempngboxlist.append(objbox)
							continue
				elif '64_CONTROLPADS'  in imgpath:
					if ymid < 1050:
						if xwidth >= 35 or yheight >= 35:
							boxlist.append(objbox)
							continue
						if xwidth > 15 or yheight > 15:
							tempngboxlist.append(objbox)
							continue
				else:
					if xwidth >= 35 or yheight >= 35:
						boxlist.append(objbox)
						continue
					if xwidth > 15 or yheight > 15:
						tempngboxlist.append(objbox)
						continue

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist 

		if CheckDefectSize(boxlist,tempngboxlist):
			return {},mdrate,[]

		if len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	return {},mdrate,[]

def WAVEGUIDE(imgpath,allngboxlist,output_dict,mdrate,allngscorelist):
	boxlist = []
	tempngboxlist = []

	idx = 0
	for objbox in allngboxlist:
		score = allngscorelist[idx]
		idx = idx + 1

		ymin = int(objbox[0])
		xmin = int(objbox[1])
		ymax = int(objbox[2])
		xmax = int(objbox[3])
		xmid = (xmin+xmax)/2
		ymid = (ymin+ymax)/2
		xwidth,yheight = rawWH(objbox)
		if '73_WAVEGUIDE' in imgpath or '75_WAVEGUIDE' in imgpath or '77_WAVEGUIDE' in imgpath :
			if xmid < 480 or xmid > 760:
				if ymid < 450:
					boxlist.append(objbox)
					continue
		elif '74_WAVEGUIDE' in imgpath or '76_WAVEGUIDE' in imgpath:
			if (xmid > 380 and xmid < 860) or xmid > 1150:
				if ymid < 450:
					boxlist.append(objbox)
					continue
		elif '78_WAVEGUIDE' in imgpath:
			if  xmid < 500:
				if ymid < 450:
					boxlist.append(objbox)
					continue
		elif '79_WAVEGUIDE' in imgpath:
			if xmid > 260 and xmid < 960:
				if ymid < 450:
					boxlist.append(objbox)
					continue
		else:
			if ymid < 450:
				boxlist.append(objbox)
				continue

	if len(boxlist) > 0:
		return output_dict,mdrate,boxlist

	return {},mdrate,[]

def HEATER(imgpath,allngboxlist,output_dict,mdrate,allngscorelist,allwdboxlist):
	sumcnt = 0
	sumymin = 0
	sumymax = 0
	avgwdymin = -1
	avgwdymax = -1
	leftwd = 1280
	rightwd = 0

	for wdbox in allwdboxlist:
		sumymin = sumymin+int(wdbox[0])
		sumymax = sumymax+int(wdbox[2])
		sumcnt = sumcnt+1
		if int(wdbox[1]) < leftwd:
			leftwd =  int(wdbox[1])
		if int(wdbox[3]) > rightwd:
			rightwd = int(wdbox[3])

	if sumcnt != 0:
		avgwdymin = int(sumymin/sumcnt)
		avgwdymax = int(sumymax/sumcnt)

		boxlist = []
		tempngboxlist = []
		idx = 0
		for objbox in allngboxlist:
			score = allngscorelist[idx]
			idx = idx + 1

			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2
			xwidth,yheight = rawWH(objbox)

			if leftwd < 600:
				rightwd = leftwd + 850
			else:
				leftwd = rightwd -850

			if xmid > rightwd-370 and xmid < rightwd-320  and ymid > avgwdymin-300 and ymid < avgwdymin-110 and yheight >= 70 and yheight < 90:
				continue

			if xmid > rightwd-400 and xmid < rightwd-70  and ymid > avgwdymax + 100 and ymid < avgwdymax + 160 and xwidth >= 100 and yheight < 30:
				continue

			if xmid > rightwd-30 and xmid < rightwd+20  and ymid > avgwdymin - 130 and ymid < avgwdymin - 20 and xwidth > 54 and yheight > 54:
				continue

			if '80_HEATER' in imgpath:
				if xmid < leftwd + 110 and ymid > avgwdymax + 120:
					continue

			#in wd
			if xmid > leftwd-10 and xmid < leftwd+170 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
				if xwidth > 20 and yheight > 20:
					boxlist.append(objbox)
					continue
				if xwidth > 29 or yheight > 29:
					boxlist.append(objbox)
					continue
				if xwidth > 15 or yheight > 15:
					tempngboxlist.append(objbox)
					continue

			if xmid > rightwd-170 and xmid < rightwd+10 and ymid > avgwdymin-10 and ymid < avgwdymax+10:
				if xwidth > 20 and yheight > 20:
					boxlist.append(objbox)
					continue
				if xwidth > 29 or yheight > 29:
					boxlist.append(objbox)
					continue
				if xwidth > 15 or yheight > 15:
					tempngboxlist.append(objbox)
					continue

			#in middle
			if xmid > leftwd-30 and xmid < rightwd+30:
				if xwidth > 39 and yheight > 39:
					boxlist.append(objbox)
					continue
				if xwidth > 60 or yheight > 60:
					boxlist.append(objbox)
					continue
				if xwidth > 30 or yheight > 30:
					tempngboxlist.append(objbox)
					continue

			#in bottom
			if ymid > avgwdymax + 120:
				if xwidth > 39 and yheight > 39:
					boxlist.append(objbox)
					continue
				if xwidth > 60 or yheight > 60:
					boxlist.append(objbox)
					continue
				if xwidth > 30 or yheight > 30:
					tempngboxlist.append(objbox)
					continue

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist

		if CheckDefectSize(boxlist,tempngboxlist,60):
			return {},mdrate,[]

		if len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist,80,160):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	else:
		boxlist = []
		tempngboxlist = []
		idx = 0
		for objbox in allngboxlist:
			score = allngscorelist[idx]
			idx = idx + 1

			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2
			xwidth,yheight = rawWH(objbox)

			if xwidth > 39 and yheight > 39:
				boxlist.append(objbox)
				continue
			if xwidth > 60 or yheight > 60:
				boxlist.append(objbox)
				continue
			if xwidth > 30 or yheight > 30:
				tempngboxlist.append(objbox)
				continue

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist

		if CheckDefectSize(boxlist,tempngboxlist,60):
			return {},mdrate,[]

		if len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist,80,160):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	return {},mdrate,[]


def GetOutPutDict(param,allscorelist1,output_dict1,allscorelist2,output_dict2,allscorelist3,output_dict3,allngboxlist,allpdboxlist,allwdboxlist,allngscorelist):
	
	output_dict = {}

	max1 = 0.0
	max2 = 0.0
	max3 = 0.0
	mdrate = 0

	if len(allscorelist1) > 0:
		max1 = max(allscorelist1)
		mdrate = mdrate+1
	if len(allscorelist2) > 0:
		max2 = max(allscorelist2)
		mdrate = mdrate+1
	if len(allscorelist3) > 0:
		max3 = max(allscorelist3)
		mdrate = mdrate+1

	if max1 != 0 and max1 >= max2 and max1 >= max3:
		output_dict = output_dict1
	elif max2 != 0 and max2 >= max1 and max2 >= max3:
		output_dict = output_dict2
	elif max3 != 0 and max3 >= max1 and max3 >= max2:
		output_dict = output_dict3

	if mdrate == 0:
		return {},mdrate,[]


	imgpath = param.rawpath.upper()

	
	if '_CONTROLPADS_' in imgpath:
		return CONTROLPADS(imgpath,allngboxlist,output_dict,mdrate,allngscorelist,allpdboxlist)
	elif '_AROUNDWAVEGUIDE_' in imgpath:
		return AROUNDWAVEGUIDE(imgpath,allngboxlist,output_dict,mdrate,allngscorelist)
	elif '_METALTRACE_' in imgpath:
		return METALTRACE(imgpath,allngboxlist,output_dict,mdrate,allngscorelist)
	elif '19_MODULATOR_'  in imgpath or '20_MODULATOR_'  in imgpath or '21_MODULATOR_'  in imgpath or '22_MODULATOR_'  in imgpath or '23_MODULATOR_'  in imgpath or '24_MODULATOR_'  in imgpath or '25_MODULATOR_'  in imgpath or '26_MODULATOR_'  in imgpath:
		return MODULATORHEATER(imgpath,allngboxlist,output_dict,mdrate,allngscorelist,allwdboxlist)
	elif '_MODULATOR_'  in imgpath:
		return MODULATOR(imgpath,allngboxlist,output_dict,mdrate,allngscorelist)
	elif '_MODULATORPADS_' in imgpath :
		return MODULATORPADS(imgpath,allngboxlist,output_dict,mdrate,allngscorelist,allpdboxlist)
	elif '_WAVEGUIDE_' in imgpath:
		return WAVEGUIDE(imgpath,allngboxlist,output_dict,mdrate,allngscorelist)
	elif '_HEATER_' in imgpath:
		return HEATER(imgpath,allngboxlist,output_dict,mdrate,allngscorelist,allwdboxlist)

	return {},mdrate,[]

def MatchBox(objbox,output_dict,param):
	for i in range(100):
		if float(output_dict['detection_scores'][0][i]) >= param.score:
			clsidx = int(output_dict['detection_classes'][0][i]) - 1
			if clsidx == 0:
				bbox = output_dict['detection_boxes'][0][i]
				if (int(objbox[0]) == int(bbox[0])) and (int(objbox[1]) == int(bbox[1])) and (int(objbox[2]) == int(bbox[2])) and (int(objbox[3]) == int(bbox[3])):
					return True
	return False


def GetMatchOutputDict(boxlist,output_dict,output_dict1,output_dict2,output_dict3,param):
	for objbox in boxlist:
		if MatchBox(objbox,output_dict,param):
			return output_dict
		if MatchBox(objbox,output_dict1,param):
			return output_dict1
		if MatchBox(objbox,output_dict2,param):
			return output_dict2
		if MatchBox(objbox,output_dict3,param):
			return output_dict3
	return output_dict


def run_model(param):
	print("   \r\n")
	print(param.rawpath)

	try:
		gamma = 0.9
		upperpath = param.rawpath.upper()
		if '_WAVEGUIDE_' in upperpath:
			param.score = 0.45
			gamma = 0.7

		newimgpath = param.rawpath.replace(".jpg","_0.jpg")
		cimgx = cv2.imread(param.rawpath,cv2.IMREAD_COLOR)
		hg,wd,ch = cimgx.shape
		cimg = cv2.resize(cimgx,(WIDTH,HIGH))

		if hg != 1280:
			cimg = CLAHE(cimg,2.1,16)
			cimg = gamma_correction_lab(cimg,gamma)
		else:
			cimgx = cv2.resize(cimgx,(2448,2048))

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

		allpdboxlist80 = []
		allpdboxlist90= []
		allwdboxlist80 = []
		allwdboxlist90 = []
		allngboxlist = []
		allngscorelist= []


		allscorelist1 = GetObjectList(output_dict1,param.score,allngboxlist,allpdboxlist80,allpdboxlist90,allwdboxlist80,allwdboxlist90,allngscorelist)
		allscorelist2 = GetObjectList(output_dict2,param.score,allngboxlist,allpdboxlist80,allpdboxlist90,allwdboxlist80,allwdboxlist90,allngscorelist)
		allscorelist3 = GetObjectList(output_dict3,param.score,allngboxlist,allpdboxlist80,allpdboxlist90,allwdboxlist80,allwdboxlist90,allngscorelist)
		
		allwdboxlist = allwdboxlist80
		if len(allwdboxlist90) > 0:
			allwdboxlist = allwdboxlist90
		
		allpdboxlist = allpdboxlist80
		if len(allpdboxlist90) > 0:
			allpdboxlist = allpdboxlist90

		output_dict,mdrate,boxlist = GetOutPutDict(param,allscorelist1,output_dict1,allscorelist2,output_dict2,allscorelist3,output_dict3,allngboxlist,allpdboxlist,allwdboxlist,allngscorelist)

		if 'detection_scores' in output_dict and len(boxlist) > 0:
			output_dict = GetMatchOutputDict(boxlist,output_dict,output_dict1,output_dict2,output_dict3,param)

		ngsubfix = '.jpg'
		if 'detection_scores' in output_dict:
			for i in range(100):
				if float(output_dict['detection_scores'][0][i]) >= param.score:
					clsidx = int(output_dict['detection_classes'][0][i]) - 1
					if clsidx == 0:
						ngsubfix = '_NG.jpg'
					color = param.colors[clsidx%30]
					tscore = round(100.0*float(output_dict['detection_scores'][0][i]),2)
					bbox = output_dict['detection_boxes'][0][i]
					drawtangle2(bbox,clsidx,cimgx,color,tscore,param)

		try:
			os.remove(newimgpath)
		except:
			print('fail to remove file')


		ngsubfix = '_'+str(mdrate)+ngsubfix
		cv2.imwrite(param.rawpath.replace("test","testout").replace('.jpg',ngsubfix),cimgx)

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
	with tf.device('/device:GPU:0'):
		try:
			paramlist = []
			score = 0.5
			colors = random_colors(30)

			print("loading test data..............")

			data_root = pathlib.Path('./mydata/trainsrcdata/ORION/test2p2')
			all_image_paths = list(data_root.glob('*'))
			all_image_paths = [str(path) for path in all_image_paths]

			print("loading model files............")

			export_dir = './mydata/trainmodels/ORION/exported_model_OR_8359xxx1'

			imported = tf.saved_model.load(export_dir)
			model_fn = imported.signatures['serving_default']

			export_dir2 = './mydata/trainmodels/ORION/exported_model_OR_8364xxx'
			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_OR_8339xxx'
			# export_dir2 = './mydata/trainmodels/PICDIE/exported_model_OR_8339xxx0'

			imported2 = tf.saved_model.load(export_dir2)
			model_fn2 = imported2.signatures['serving_default']



			export_dir3 = './mydata/trainmodels/ORION/exported_model_OR_8352xxx0'

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

if __name__ == "__main__":
	MainLoop()
