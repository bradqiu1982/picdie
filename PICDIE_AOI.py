import argparse
import os
import traceback
import sys
import logging
logging.getLogger('absl').setLevel('ERROR')
import pyodbc
import json
import tensorflow as tf
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import io
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
import pathlib
import cv2
import copy
import uuid
import base64
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
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timedelta
import gc

HIGH = 1280
WIDTH = 1280
cache = dict()
Ascore = 0.5
pathdict = {}
# DBCONNECTSTR = 'mongodb://NPI:NPI%40NPI@cnwx-engsys02:27017/?directConnection=true&authSource=NPITrace'
DBCONNECTSTR = 'mongodb://NPI:NPI%40NPI@cnwx-sdwww03:27017/?directConnection=true&authSource=NPITrace'

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


def  getPICDIEModel():
	PICDIEMODEL='PICDIEMODEL'
	if PICDIEMODEL not in cache:
		export_dir = './AOI/PIC_AOI/exported_model_OR_8542xxxxx0'
		# export_dir = './AOI/PIC_AOI/exported_model_OR_8526xxxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_OR_8484xxxx1'
		# export_dir = './AOI/PIC_AOI/exported_model_B91_8218xxxxxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_B91_7975xxxxxx0'
		# export_dir = './AOI/PIC_AOI/exported_model_B91_8084xxxxxx'


		imported = tf.saved_model.load(export_dir)
		model_fn = imported.signatures['serving_default']
		cache[PICDIEMODEL] = model_fn
		return model_fn
	else:
		return cache[PICDIEMODEL]

def  getPICDIEModel2():
	PICDIEMODEL='PICDIEMODEL2'
	if PICDIEMODEL not in cache:
		export_dir = './AOI/PIC_AOI/exported_model_OR_8495xxxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_OR_8528xxxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_OR_8529xxxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_OR_8487xxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_B91_8232xxxxxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_B91_8009xxxxxx'
		#export_dir = './AOI/PIC_AOI/exported_model_B91_7925xxxxx'

	
		imported = tf.saved_model.load(export_dir)
		model_fn2 = imported.signatures['serving_default']
		cache[PICDIEMODEL] = model_fn2
		return model_fn2
	else:
		return cache[PICDIEMODEL]

def  getPICDIEModel3():
	PICDIEMODEL='PICDIEMODEL3'
	if PICDIEMODEL not in cache:
		export_dir = './AOI/PIC_AOI/exported_model_OR_8552xxxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_OR_8538xxxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_OR_850xxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_B91_8236xxxxxxx0'
		# export_dir = './AOI/PIC_AOI/exported_model_B91_8003xxxxxx'
		# export_dir = './AOI/PIC_AOI/exported_model_B91_8012xxxxxx'


		imported = tf.saved_model.load(export_dir)
		model_fn3 = imported.signatures['serving_default']
		cache[PICDIEMODEL] = model_fn3
		return model_fn3
	else:
		return cache[PICDIEMODEL]


class PICDIEITEM:
	def __init__(self,aoikey,pj,wafer,cellpos,rawpath,tobepath,colors,score,model_fn,model_fn2,model_fn3,uptime):
		self.aoikey = aoikey
		self.pj = pj
		self.wafer = wafer
		self.cellpos = cellpos
		self.rawpath = rawpath.lower().replace('\\wux-fs','\\datacom-fs')
		self.tobepath = tobepath
		self.colors = colors
		self.score = score
		self.model_fn = model_fn
		self.model_fn2 = model_fn2
		self.model_fn3 = model_fn3
		self.uptime = uptime

		self.newimgpath = ''	
		self.cimgx = None
		self.img_tensor = None


class AOIRESTITEM:
	def __init__(self,aoikey,pj,wafer,cellpos,tobepath,analyzepath,aoirest,maxscore,MDRate):
		self.aoikey = aoikey
		self.pj = pj
		self.wafer = wafer
		self.cellpos = cellpos
		self.tobepath = tobepath
		self.analyzepath = analyzepath
		self.aoirest = aoirest
		now = datetime.utcnow()
		self.analyzedtime = now
		self.maxscore = maxscore
		self.MDRate = MDRate

def getRunID(cellpos,modnum):
	idx = 0
	try:
		idx = int(cellpos)%modnum
	except:
		print('exception cellpos in runid:'+cellpos)
		return 0
	return idx

def reverse_num(num):
	return int(str(num)[::-1])

def getRunID2(cellpos,modnum2):
	idx = 0
	try:
		revnum = reverse_num(int(cellpos))
		idx = revnum%modnum2
	except:
		print('exception cellpos in runid:'+cellpos)
		return 0
	return idx


def GetAOIItems(modnum,myrunid,modnum2,myrunid2):
	colors = random_colors(30)
	model_fn = getPICDIEModel()
	model_fn2 = getPICDIEModel2()
	# model_fn3 = getPICDIEModel3()
	model_fn3 = None

	AOIItemList = []
	try:
		start_date = datetime.utcnow() - timedelta(days=15)
		# start_date = datetime(2025, 6, 30, 0, 0, 0, 0)
		myclient = pymongo.MongoClient(DBCONNECTSTR)
		mydb = myclient["NPITrace"]

		piccol = mydb["PICVM"]
		query = {'$and':[{'Project':'2X400G_FR4_BBLC_SiPh'},{'TestTime':{'$gte':start_date}},{'DieImgCnt':{'$gte':94}},{'SKAnalyzed':0}]}
		waferlist = piccol.distinct('Wafer',query)

		aoicol = mydb["PICDIEAOI"]
		query = {'$and':[{'Project':'2X400G_FR4_BBLC_SiPh'},{'Wafer':{'$in':waferlist}},{'Analyzed':0}]}
		field = {'_id':1,'Project':1,'Wafer':1,'CellPos':1,'RawPath':1,'ToBePath':1,'UpdateTime':1}
		for x in aoicol.find(query,field):
			if '.JPG' in x['RawPath'].upper() or '.PNG' in x['RawPath'].upper():
				cellpos = x['CellPos']
				runid = getRunID(cellpos,modnum)
				if runid != myrunid:
					continue
				if len(x['Wafer']) > 14 or len(x['Wafer']) < 12:
					continue

				if modnum2 != 0:
					runid2 = getRunID2(cellpos,modnum2)
					if runid2 != myrunid2:
						continue

				item = PICDIEITEM(str(x['_id']),x['Project'],x['Wafer'],cellpos,x['RawPath'],x['ToBePath'],colors,Ascore,model_fn,model_fn2,model_fn3,x['UpdateTime'])
				AOIItemList.append(item)
	except:
		print('a database except happend................')
		time.sleep(5)
		
	AOIItemList.sort(key=lambda x: x.uptime)
	if len(AOIItemList) > 30000:
		AOIItemList = AOIItemList[:30000]
	return AOIItemList


def drawtangle2(box,cls,cimg,color,tscore):

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
	cv2.putText(cimg,ref_dict[int(cls)]+','+str(xwidth)+','+str(yheight),(xmark,ymark),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)


# def drawtangle2(box,cls,cimg,color,tscore):
# 	ymin = int(box[0])
# 	xmin = int(box[1])
# 	ymax = int(box[2])
# 	xmax = int(box[3])

# 	cv2.rectangle(cimg,(xmin-1,ymin-1),(xmax+1,ymax+1),(0,0,255),2)

# 	ref_dict = {}
# 	ref_dict[0] = 'ng'
# 	ref_dict[1] = 'pd'
# 	ref_dict[2] = 'wd'
	
# 	if xmax < 300:
# 		cv2.putText(cimg,ref_dict[int(cls)],(xmin+20,ymax+36),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
# 		cv2.putText(cimg,str(tscore)+','+str(xmax-xmin)+','+str(ymax-ymin),(xmin+60,ymax+36),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)
# 	else:
# 		cv2.putText(cimg,ref_dict[int(cls)],(xmin-160,ymax+36),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
# 		cv2.putText(cimg,str(tscore)+','+str(xmax-xmin)+','+str(ymax-ymin),(xmin-120,ymax+36),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)


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


def SaveTobeImg(cimg,param):
	fps = param.rawpath.split("\\")
	if(len(fps) < 2):
		fps = filepath.split("/")
	fn = fps[len(fps)-1]

	# basepath = '\\\\WUX-FS\\Datacom_Test_Data02\\WUXI_AI\\AOI\\PICAOI\\2X400G_FR4_BBLC_SiPh\\'
	# basepath = '\\\\WUX-FS\\Datacom_Test_Data03\\WUXI_AI02\\AOI\\PICAOI\\2X400G_FR4_BBLC_SiPh\\'
	basepath = '\\\\cnwx-cifs\\Datacom_Test_Data03\\WUXI_AI02\\AOI\\PICAOI\\2X400G_FR4_BBLC_SiPh\\'

	waferpath = basepath+param.wafer
	try:
		if waferpath not in pathdict:
			if os.path.exists(waferpath):
				x= 0
			else:
				os.makedirs(waferpath)
			pathdict[waferpath] = True
	except:
		return ''

	rawpath = waferpath+'\\raw'
	try:
		if rawpath not in pathdict:
			if os.path.exists(rawpath):
				x= 0
			else:
				os.makedirs(rawpath)
			pathdict[rawpath] = True
	except:
		return ''

	# cellpospath = rawpath+'\\'+param.cellpos
	# try:
	# 	if cellpospath not in pathdict:
	# 		if os.path.exists(cellpospath):
	# 			x= 0
	# 		else:
	# 			os.makedirs(cellpospath)
	# 		pathdict[cellpospath] = True
	# except:
	# 	return ''

	try:
		newimgpath = rawpath+'\\'+param.cellpos+'_'+fn
		cv2.imwrite(newimgpath,cimg)
		return newimgpath
	except:
		exception_message = sys.exc_info()[1]
		print(str(exception_message))
		return ''

def SaveAnalyzedImg(cimg,param,NGsubfix):
	fps = param.rawpath.split("\\")
	if(len(fps) < 2):
		fps = filepath.split("/")
	fn = fps[len(fps)-1]

	# basepath = '\\\\WUX-FS\\Datacom_Test_Data02\\WUXI_AI\\AOI\\PICAOI\\2X400G_FR4_BBLC_SiPh\\'
	# basepath = '\\\\WUX-FS\\Datacom_Test_Data03\\WUXI_AI02\\AOI\\PICAOI\\2X400G_FR4_BBLC_SiPh\\'
	basepath = '\\\\cnwx-cifs\\Datacom_Test_Data03\\WUXI_AI02\\AOI\\PICAOI\\2X400G_FR4_BBLC_SiPh\\'
	
	waferpath = basepath+param.wafer
	try:
		if waferpath not in pathdict:
			if os.path.exists(waferpath):
				x= 0
			else:
				os.makedirs(waferpath)
			pathdict[waferpath] = True
	except:
		return ''

	rawpath = waferpath+'\\analyzed'
	try:
		if rawpath not in pathdict:
			if os.path.exists(rawpath):
				x= 0
			else:
				os.makedirs(rawpath)
			pathdict[rawpath] = True
	except:
		return ''

	# cellpospath = rawpath+'\\'+param.cellpos
	# try:
	# 	if cellpospath not in pathdict:
	# 		if os.path.exists(cellpospath):
	# 			x= 0
	# 		else:
	# 			os.makedirs(cellpospath)
	# 		pathdict[cellpospath] = True
	# except:
	# 	return ''

	try:
		newimgpath = rawpath+'\\'+param.cellpos+'_'+fn.lower()
		newimgpath = newimgpath.replace(".jpg",NGsubfix)
		cv2.imwrite(newimgpath,cimg)
		return newimgpath
	except:
		exception_message = sys.exc_info()[1]
		print(str(exception_message))
		return ''


def CheckPadS(objbox,allpdboxlist):
	ymin = int(objbox[0])
	xmin = int(objbox[1])
	ymax = int(objbox[2])
	xmax = int(objbox[3])

	xwidth = xmax-xmin
	yheight = ymax-ymin
	xmid = (xmin+xmax)/2
	ymid = (ymin+ymax)/2

	for pdbox in allpdboxlist:
		pdymin = int(pdbox[0])
		pdxmin = int(pdbox[1])
		pdymax = int(pdbox[2])
		pdxmax = int(pdbox[3])
		pdxwidth = pdxmax-pdxmin
		pdyheight = pdymax-pdymin

		if xmid > pdxmin and xmid < pdxmax and ymid > pdymin and ymid < pdymax:
			if xwidth*yheight >= 0.3*pdxwidth*pdyheight:
				return True
			else:
				return False
	return True

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

def checkboxdist(boxlist,distthold1=60,distthold2=140):
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

			p = [xmin1,ymin1]
			q = [xmax2,ymax2]
			dist1 = math.dist(p,q)

			p = [xmax1,ymin1]
			q = [xmin2,ymax2]
			dist2 = math.dist(p,q)

			if (dist1 > distthold1 and dist1 < distthold2) or (dist2 > distthold1 and dist2 < distthold2) :
				return True
	return False

def AroundWaveGuide(imgpath,allngboxlist,output_dict,mdrate):
	boxlist = []
	tempngboxlist = []

	for objbox in allngboxlist:
		ymin = int(objbox[0])
		xmin = int(objbox[1])
		ymax = int(objbox[2])
		xmax = int(objbox[3])
		xwidth = xmax-xmin
		yheight = ymax-ymin
		xmid = (xmin+xmax)/2
		ymid = (ymin+ymax)/2

		if '2_AROUNDWAVEGUIDE' in imgpath or '3_AROUNDWAVEGUIDE' in imgpath:
			if xmax < 380 and yheight > 120:
				continue

		if (ymin > 220 and ymin < 280) or (ymax > 220 and ymax < 280) or (ymid > 220 and ymid < 280):
			if xwidth >= 35 or yheight >= 35:
				boxlist.append(objbox)
	if len(boxlist) > 0:
		return output_dict,mdrate,boxlist
	else:
		return {},mdrate,[]

	return {},mdrate,[]

def MetalTrace1(allngboxlist,output_dict,mdrate):
	boxlist = []
	tempngboxlist = []

	for objbox in allngboxlist:
		ymin = int(objbox[0])
		xmin = int(objbox[1])
		ymax = int(objbox[2])
		xmax = int(objbox[3])
		xwidth = xmax-xmin
		yheight = ymax-ymin
		xmid = (xmin+xmax)/2
		ymid = (ymin+ymax)/2

		if (ymin > 600 and ymin < 1010) or (ymax > 600 and ymax < 1010) or (ymid > 600 and ymid < 1010):
			if xwidth >= 45 or yheight >= 45:
				boxlist.append(objbox)
			tempngboxlist.append(objbox)
			continue

	if CheckUpDownTraceSize(boxlist,tempngboxlist,600,1010):
		return {},mdrate,[]

	if len(boxlist) > 0:
		return output_dict,mdrate,boxlist
	elif len(tempngboxlist) > 0:
		if checkboxdist(tempngboxlist):
			return output_dict,mdrate,boxlist
	else:
		return {},mdrate,[]

	return {},mdrate,[]

#'21_METALTRACE' in imgpath or '22_METALTRACE' in imgpath or '23_METALTRACE' in imgpath or '24_METALTRACE' in imgpath or '25_METALTRACE' in imgpath:
def MetalTrace2(imgpath,allngboxlist,allwdboxlist,output_dict,mdrate):
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

	if avgwdymin != -1:
		avbymin1 = avgwdymin-360
		avbymax1 = avgwdymin-175
		avbymin2 = avgwdymax+85
		# avbymax2 = avgwdymax+275
		avbymax2 = 1280

		matchlinearea = False

		for objbox in allngboxlist:
			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if (ymin > avbymin1 and ymin < avbymax1) or (ymax > avbymin1 and ymax < avbymax1) or (ymid > avbymin1 and ymid < avbymax1):
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if (ymin > avbymin2 and ymin < avbymax2) or (ymax > avbymin2 and ymax < avbymax2) or (ymid > avbymin2 and ymid < avbymax2):
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				matchlinearea = True
				continue

			for wdbox in allwdboxlist:
				wdymin = int(wdbox[0])
				wdxmin = int(wdbox[1])
				wdymax = int(wdbox[2])
				wdxmax = int(wdbox[3])
				if((ymin > wdymin and ymin < wdymax) or (ymax > wdymin and ymax < wdymax) or (ymid > wdymin and ymid < wdymax))  and ((xmin > wdxmin and xmin < wdxmax) or (xmax > wdxmin and xmax < wdxmax) or (xmid > wdxmin and xmid < wdxmax)):
					if xwidth >= 45 or yheight >= 45:
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					break

			if '21_METALTRACE' in imgpath:
				if leftwd < 220:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and ((xmin > leftwd+30 and xmin < leftwd+400) or (xmax > leftwd+30 and xmax < leftwd+400) or (xmid > leftwd+30 and xmid < leftwd+400)):
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
				elif leftwd < 550:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and ((xmin > leftwd-320 and xmin < leftwd+50) or (xmax > leftwd-320 and xmax < leftwd+50) or (xmid > leftwd-320 and xmid < leftwd+50)):
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
			if '22_METALTRACE' in imgpath:
				if leftwd > 100 and leftwd < 500:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmin < leftwd + 50:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmax > leftwd + 820:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
				elif leftwd > 800:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmin < leftwd -740:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmax > leftwd + 50:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
			if '23_METALTRACE' in imgpath:
				if leftwd < 400:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmin < leftwd + 50:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmax > leftwd + 820:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
				elif leftwd > 800:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmin < leftwd -740:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmax > leftwd + 50:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
			if '24_METALTRACE' in imgpath:
				if leftwd < 400:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmin < leftwd + 50:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmax > leftwd + 820:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
				elif leftwd > 700:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmin < leftwd -740:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmax > leftwd + 50:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
			if '25_METALTRACE' in imgpath:
				if rightwd < 300:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmin < rightwd -35:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmax > rightwd + 740:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
				elif rightwd > 600 and rightwd < 1000:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmin < rightwd -830:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmax > rightwd - 50:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
				elif rightwd > 1000:
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmin < rightwd -1180:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and xmax > rightwd - 400:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

		if CheckUpDownTraceSize(boxlist,tempngboxlist,avgwdymax+30,1280,40):
			return {},mdrate,[]

		if CheckUpDownTraceSize(boxlist,tempngboxlist,0,avbymax1,40):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if matchlinearea:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				if checkboxdist(tempngboxlist):
					return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

		return {},mdrate,[]		
	else:
		for objbox in allngboxlist:
			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if (ymin > 250 and ymin < 400) or (ymax > 250 and ymax < 400) or (ymid > 250 and ymid < 400):
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if (ymin > 890 and ymin < 1130) or (ymax > 890 and ymax < 1130) or (ymid > 890 and ymid < 1130):
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

		return {},mdrate,[]

def MetalTrace3(imgpath,allngboxlist,allwdboxlist,output_dict,mdrate):
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

	if avgwdymin != -1:
		avbymin1 = avgwdymin-320
		avbymax1 = avgwdymin-175
		avbymin2 = avgwdymax+85
		#avbymax2 = avgwdymax+275
		avbymax2 = 1280
		matchlinearea = False

		for objbox in allngboxlist:
			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if (ymin > avbymin1 and ymin < avbymax1) or (ymax > avbymin1 and ymax < avbymax1) or (ymid > avbymin1 and ymid < avbymax1):
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if (ymin > avbymin2 and ymin < avbymax2) or (ymax > avbymin2 and ymax < avbymax2) or (ymid > avbymin2 and ymid < avbymax2):
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				matchlinearea = True
				continue

			for wdbox in allwdboxlist:
				wdymin = int(wdbox[0])
				wdxmin = int(wdbox[1])
				wdymax = int(wdbox[2])
				wdxmax = int(wdbox[3])
				if((ymin > wdymin and ymin < wdymax) or (ymax > wdymin and ymax < wdymax) or (ymid > wdymin and ymid < wdymax))   and ((xmin > wdxmin and xmin < wdxmax) or (xmax > wdxmin and xmax < wdxmax) or (xmid > wdxmin and xmid < wdxmax)):
					if xwidth >= 45 or yheight >= 45:
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					break

			if '20_METALTRACE' in imgpath:
				if leftwd < 280:
					# if  ymax > avbymax2-20 and xmin < leftwd -70:
					# 	if xwidth >= 45 or yheight >= 45:
					# 		boxlist.append(objbox)
					# 	tempngboxlist.append(objbox)
					# 	continue

					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and ((xmin > leftwd+30 and xmin < leftwd+400) or (xmax > leftwd+30 and xmax < leftwd+400) or (xmid > leftwd+30 and xmid < leftwd+400)):
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

				elif leftwd > 300 and leftwd < 700:
					# if  ymax > avbymax2-20 and xmin < leftwd -420:
					# 	if xwidth >= 45 or yheight >= 45:
					# 		boxlist.append(objbox)
					# 	tempngboxlist.append(objbox)
					# 	continue

					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and ((xmin > leftwd-320 and xmin < leftwd+50) or (xmax > leftwd-320 and xmax < leftwd+50) or (xmid > leftwd-320 and xmid < leftwd+50)):
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

			elif '26_METALTRACE' in imgpath:
				if leftwd < 900:
					if  ymax > avbymax2-20 and xmax > leftwd+520:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

					if  ymax > avbymin1 and xmax > leftwd +560:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and ((xmin > leftwd+30 and xmin < leftwd+400) or (xmax > leftwd+30 and xmax < leftwd+400) or (xmid > leftwd+30 and xmid < leftwd+400)):
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

				elif leftwd > 920:
					if  ymax > avbymax2-20 and xmax > leftwd+160:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

					if  ymax > avbymin1 and xmax > rightwd +200:
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

					if ((ymax >  avgwdymax+30) and (ymax < avgwdymax+120)) and ((xmin > leftwd-320 and xmin < leftwd+50) or (xmax > leftwd-320 and xmax < leftwd+50) or (xmid > leftwd-320 and xmid < leftwd+50)):
						if xwidth >= 45 or yheight >= 45:
							boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

		if CheckUpDownTraceSize(boxlist,tempngboxlist,avgwdymax+30,1280):
			return {},mdrate,[]

		if CheckUpDownTraceSize(boxlist,tempngboxlist,0,avbymax1):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if matchlinearea:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				if checkboxdist(tempngboxlist):
					return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

		return {},mdrate,[]
	else:
		for objbox in allngboxlist:
			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if (ymin > 250 and ymin < 400) or (ymax > 250 and ymax < 400) or (ymid > 250 and ymid < 400):
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if (ymin > 890 and ymin < 1130) or (ymax > 890 and ymax < 1130) or (ymid > 890 and ymid < 1130):
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if '20_METALTRACE' in imgpath:
				if  ymax > 1130 and xmin < 40:
					if xwidth >= 45 or yheight >= 45:
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			elif '26_METALTRACE' in imgpath:
				if  ymax > 1130 and xmax > 1200:
					if xwidth >= 45 or yheight >= 45:
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if  ymax > 400 and xmax > 1240:
					if xwidth >= 45 or yheight >= 45:
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

		return {},mdrate,[]


def Modulator(imgpath,allngboxlist,output_dict,mdrate,allngscorelist):
	if '29_MODULATOR' in imgpath:
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
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if (xwidth <=14 or (xwidth >= 1.8*yheight and yheight <= 30)) and ymid > 1100:
				continue

			if xmax > 1265 and xwidth <= 20 and yheight >= 4.5*xwidth and score < 0.9:
				continue

			if xmax > 1180:
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if ymax > 1010 and ((xmin > 770 and xmin < 1060) or (xmax > 770 and xmax < 1060) or (xmid > 770 and xmid < 1060)):
				if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

		if CheckModulatorDefectSize(boxlist,tempngboxlist,20):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]
	elif '30_MODULATOR' in imgpath or '31_MODULATOR' in imgpath or '32_MODULATOR' in imgpath or '33_MODULATOR' in imgpath or '34_MODULATOR' in imgpath:
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
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if (xwidth <=14 or (xwidth >= 1.8*yheight and yheight <= 30)) and ymid > 1100:
				continue

			if xmax > 1265 and xwidth <= 32 and yheight >= 7*xwidth and score < 0.9:
				continue

			if '30_MODULATOR' in imgpath:
				if ymax > 1010 and ((xmin > 860 and xmin < 1150) or (xmax > 860 and xmax < 1150) or (xmid > 860 and xmid < 1150)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
			elif '31_MODULATOR' in imgpath:
				if ymax > 1010 and ((xmin < 100) or ((xmin > 960 and xmin < 1240) or (xmax > 960 and xmax < 1240) or (xmid > 960 and xmid < 1240))):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
			elif '32_MODULATOR' in imgpath:
				if ymax > 1010 and (xmin < 190 or xmax > 1040):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
			elif '33_MODULATOR' in imgpath:
				if  ymax > 1010 and (xmin < 280 or xmax > 1130):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
			elif '34_MODULATOR' in imgpath:
				if ymax > 1010 and (((xmin > 80 and xmin < 370) or (xmax > 80 and xmax < 370) or (xmid > 80 and xmid < 370)) or (xmax > 1220)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

		if CheckModulatorDefectSize(boxlist,tempngboxlist,20):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]
	elif '35_MODULATOR' in imgpath:
		boxlist = []
		tempngboxlist = []

		for objbox in allngboxlist:
			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if (xwidth <=14 or (xwidth >= 1.8*yheight and yheight <= 30)) and ymid > 1100:
				continue

			if xmin < 25:
				if xmin <= 25 and yheight >= 70 and yheight < 100 and xwidth >= 30 and xwidth < 55 and ymin > 150 and ymin < 240:
					continue

				if xmin <= 25 and yheight >= 105 and yheight < 130 and xwidth >= 35 and xwidth < 50 and ymin < 10:
					continue

				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if ymax > 1010 and ((xmin > 180 and xmin < 470) or (xmax > 180 and xmax < 470) or (xmid > 180 and xmid < 470)):
				if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue
		
		if CheckModulatorDefectSize(boxlist,tempngboxlist,20):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	elif '38_MODULATOR' in imgpath or '53_MODULATOR' in imgpath or '56_MODULATOR' in imgpath or '71_MODULATOR' in imgpath:
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
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if xmin < 15 and xwidth <= 20 and yheight >= 4.5*xwidth and score < 0.9:
				continue

			if ymax > 1270 and xwidth >= 3.5*yheight and yheight < 20 and xmin > 250 and xmin < 400 and score < 0.9:
				continue

			if'38_MODULATOR' in imgpath and ((xwidth <=14 or (xwidth >= 1.8*yheight and yheight <= 30)) and ymid < 100):
				continue

			if xmin < 40:
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if (xmin > 170 and xmin < 460) or (xmax > 170 and xmax < 460) or (xmid > 170 and xmid < 460):
				if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if (xmin > 170 and xmin < 490) or (xmax > 170 and xmax < 490) or (xmid > 170 and xmid < 490):
				if (xwidth >= 45 and yheight >= 45):
					boxlist.append(objbox)
				continue

		if CheckModulatorDefectSize(boxlist,tempngboxlist,20):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	elif '39_MODULATOR' in imgpath or '52_MODULATOR' in imgpath or '57_MODULATOR' in imgpath or '70_MODULATOR' in imgpath:
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
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if xmax > 1265 and xwidth <= 20 and yheight >= 4.5*xwidth and score < 0.9:
				continue

			if'39_MODULATOR' in imgpath and ((xwidth <=14 or (xwidth >= 1.8*yheight and yheight <= 30)) and ymid < 100):
				continue

			if (xmax > 1210) or ((xmin > 50 and xmin < 380) or (xmax > 50 and xmax < 380) or (xmid > 50 and xmid < 380)):
				if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue
			elif (xmax > 1190) or ((xmin > 30 and xmin < 400) or (xmax > 30 and xmax < 400) or (xmid > 30 and xmid < 400)): 
				if (xwidth >= 45 and yheight >= 45) or (xwidth > 30 and xwidth*yheight > 2000):
					boxlist.append(objbox)
				continue

		if CheckModulatorDefectSize(boxlist,tempngboxlist,20):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]
	elif '40_MODULATOR' in imgpath or '51_MODULATOR' in imgpath or '58_MODULATOR' in imgpath or '69_MODULATOR' in imgpath \
		or '41_MODULATOR' in imgpath or '50_MODULATOR' in imgpath or '59_MODULATOR' in imgpath or '68_MODULATOR' in imgpath \
		or '42_MODULATOR' in imgpath or '49_MODULATOR' in imgpath or '60_MODULATOR' in imgpath or '67_MODULATOR' in imgpath:
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
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if ymax > 1270 and xwidth >= 2.7*yheight and yheight < 20 and xmin > 1000 and xmin < 1200 and score < 0.9:
				continue

			if ymax > 1270 and xwidth >= 4*yheight and yheight < 20 and xmin < 100 and score < 0.9:
				continue

			if xmax > 1265 and xwidth <= 20 and yheight >= 4.5*xwidth and score < 0.9:
				continue

			if ('40_MODULATOR' in imgpath or '41_MODULATOR' in imgpath or '42_MODULATOR' in imgpath) and ((xwidth <=14 or (xwidth >= 1.8*yheight and yheight <= 30)) and ymid < 100):
				continue

			if '40_MODULATOR' in imgpath or '51_MODULATOR' in imgpath or '58_MODULATOR' in imgpath or '69_MODULATOR' in imgpath:
				if xmin < 280 or xmax > 1070:
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				elif  xmin < 300 or xmax > 1130:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					continue

			elif '41_MODULATOR' in imgpath or '50_MODULATOR' in imgpath or '59_MODULATOR' in imgpath or '68_MODULATOR' in imgpath:
				if xmin < 200 or xmax > 1050:
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				elif  xmin < 220 or xmax > 1030:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					continue
			elif  '42_MODULATOR' in imgpath or '49_MODULATOR' in imgpath or '60_MODULATOR' in imgpath or '67_MODULATOR' in imgpath:
				if xmin < 100 or xmax > 870:
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
		
		if CheckModulatorDefectSize(boxlist,tempngboxlist,20):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	elif '43_MODULATOR' in imgpath or '48_MODULATOR' in imgpath or '61_MODULATOR' in imgpath or '66_MODULATOR' in imgpath:
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
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if ymax > 1270 and xwidth >= 4*yheight and yheight < 20 and xmin > 900 and xmin < 1100 and score < 0.9:
				continue

			if ('43_MODULATOR' in imgpath) and ((xwidth <=14 or (xwidth >= 1.8*yheight and yheight <= 30)) and ymid < 100):
				continue

			if (xmin > 840 and xmin < 1150) or (xmax > 840 and xmax < 1150) or (xmid > 840 and xmid < 1150):
				if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if (xmin > 840 and xmin < 1190) or (xmax > 840 and xmax < 1190) or (xmid > 840 and xmid < 1190):
				if (xwidth >= 45 and yheight >= 45):
					boxlist.append(objbox)
				continue

		if CheckModulatorDefectSize(boxlist,tempngboxlist,20):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]
	elif '44_MODULATOR' in imgpath or '47_MODULATOR' in imgpath or '62_MODULATOR' in imgpath or '65_MODULATOR' in imgpath:
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
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if xmax > 1265 and xwidth <= 22 and yheight >= 4*xwidth and score < 0.9:
				continue

			if ymax > 1270 and xwidth >= 4*yheight and yheight < 20 and xmin > 800 and xmin < 1000 and score < 0.9:
				continue

			if ('44_MODULATOR' in imgpath) and ((xwidth <=14 or (xwidth >= 1.8*yheight and yheight <= 30)) and ymid < 100):
				continue

			if xmax > 1200:
				if xwidth >= 45 or yheight >= 45:
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if (xmin > 760 and xmin < 1080) or (xmax > 760 and xmax < 1080) or (xmid > 760 and xmid < 1080):
				if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue
			elif (xmin > 760 and xmin < 1090) or (xmax > 760 and xmax < 1090) or (xmid > 760 and xmid < 1090):
				if (xwidth >= 45 and yheight >= 45):
					boxlist.append(objbox)
				continue

		if CheckModulatorDefectSize(boxlist,tempngboxlist,20):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	return {},mdrate,[]

def ModulatorPads(imgpath,allngboxlist,output_dict,mdrate,allpdboxlist,allngscorelist):
	boxlist = []
	tempngboxlist = []

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

	if avgpdymin != -1:


		if '74_MODULATORPADS' in imgpath:
			markpos = -1
			for pdbox in allpdboxlist:
				pdwidth = int(pdbox[3])-int(pdbox[1])
				pdxmin = int(pdbox[1])
				pdxmax = int(pdbox[3])
				if pdxmin < 150:
					markpos = pdxmax+345
					break
				elif pdxmax > 1230:
					markpos = pdxmin-585
					break
				elif pdwidth > 180 and pdxmax > 400 and pdxmax < 800:
					markpos = pdxmax
					break
				elif pdwidth < 130 and pdxmax > 800 and pdxmax < 1100:
					markpos = pdxmax - 345
					break

			if markpos == -1:
				markpos = 600

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if ymin >= avgpdymin and ymax < avgpdymax and ((xwidth <= 46 and yheight <= 46) or xwidth*yheight < 1500):
					continue
				if ymin >= avgpdymin and ymax < avgpdymax and (xmin < 15 or xmax > 1265) and xwidth < 30:
					continue

				if ymin >= avgpdymax and int(yheight*0.5333) < 55:
					continue

				if (ymin < avgpdymin-385) and ((xmin > markpos-420 and xmin < markpos-130 ) or (xmax > markpos-420 and xmax < markpos-130) or (xmid > markpos-420 and xmid < markpos-130)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
							boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin < avgpdymin-500 and xmin < markpos - 370:
					if (xwidth >= 45 and yheight >= 45) :
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmin < markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymax > avgpdymin - 230  and ymax < avgpdymin+50 and xmin > markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymin and ymid < avgpdymax+10:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
					if xwidth > 100 and score > 0.9:
						boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymax and ymid < avgpdymax+80 and xwidth*yheight > 60*60 and score > 0.9:
					boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

		if '75_MODULATORPADS' in imgpath:
			markpos = -1
			for pdbox in allpdboxlist:
				pdwidth = int(pdbox[3])-int(pdbox[1])
				pdxmin = int(pdbox[1])
				pdxmax = int(pdbox[3])
				if pdxmin < 50:
					markpos = pdxmax+345
					break
				elif pdxmax > 1230:
					markpos = pdxmin-585
					break
				elif pdwidth > 180 and pdxmax < 800:
					markpos = pdxmax
					break
				elif pdwidth < 130 and pdxmax > 600 and pdxmax < 1000:
					markpos = pdxmax-345
					break
			if markpos == -1:
				markpos = 510

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if ymin >= avgpdymin and ymax < avgpdymax and ((xwidth <= 46 and yheight <= 46) or xwidth*yheight < 1500):
					continue
				if ymin >= avgpdymin and ymax < avgpdymax and (xmin < 15 or xmax > 1265) and xwidth < 30:
					continue

				if ymin >= avgpdymax and  int(yheight*0.5333) < 55:
					continue


				if (ymin < avgpdymin-385) and ((xmin > markpos-425 and xmin < markpos-130 ) or (xmax > markpos-425 and xmax < markpos-130) or (xmid > markpos-425 and xmid < markpos-130)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymin < avgpdymin-385) and (xmax > markpos+720):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmin < markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymax > avgpdymin - 230  and ymax < avgpdymin+50 and xmin > markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmax > markpos+590:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymin and ymid < avgpdymax+10:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
					if xwidth > 100 and score > 0.9:
						boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymax and ymid < avgpdymax+80 and xwidth*yheight > 60*60 and score > 0.9:
					boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

		if '76_MODULATORPADS' in imgpath:
			markpos = -1
			for pdbox in allpdboxlist:
				pdwidth = int(pdbox[3])-int(pdbox[1])
				pdxmin = int(pdbox[1])
				pdxmax = int(pdbox[3])
				if pdxmin < 50:
					markpos = pdxmax+345
					break
				elif pdxmax > 1100:
					markpos = pdxmin-585
					break
				elif pdwidth > 180 and pdxmax < 600:
					markpos = pdxmax
					break
				elif pdwidth < 130 and pdxmax > 500 and pdxmax < 1000:
					markpos = pdxmax-345
					break
			if markpos == -1:
				markpos = 415

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if ymin >= avgpdymin and ymax < avgpdymax and ((xwidth <= 46 and yheight <= 46) or xwidth*yheight < 1500):
					continue
				if ymin >= avgpdymin and ymax < avgpdymax and (xmin < 15 or xmax > 1265) and xwidth < 30:
					continue

				if ymin >= avgpdymax and  int(yheight*0.5333) < 55:
					continue

				if xmax > 1270 and xwidth <= 20 and yheight >= 5*xwidth and score < 0.9:
					continue

				if xmax > 1270 and xwidth <= 35 and yheight >= 6*xwidth and score < 0.9:
					continue

				if (ymin < avgpdymin-385) and ((xmin > markpos-425 and xmin < markpos-130 ) or (xmax > markpos-425 and xmax < markpos-130) or (xmid > markpos-425 and xmid < markpos-130)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
					
				if (ymin < avgpdymin-385) and (xmax > markpos+720):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmin < markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymax > avgpdymin - 230  and ymax < avgpdymin+50 and xmin > markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmax > markpos+590:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymin and ymid < avgpdymax+10:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
					if xwidth > 100 and score > 0.9:
						boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymax and ymid < avgpdymax+80 and xwidth*yheight > 60*60 and score > 0.9:
					boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

		if '77_MODULATORPADS' in imgpath:
			markpos = -1
			for pdbox in allpdboxlist:
				pdwidth = int(pdbox[3])-int(pdbox[1])
				pdxmin = int(pdbox[1])
				pdxmax = int(pdbox[3])
				if pdwidth > 180 and pdxmax < 700:
					markpos = pdxmax
					break
				elif pdwidth > 180 and pdxmax > 700:
					markpos = pdxmax-800
					break
				elif pdwidth < 130 and pdxmax > 400 and pdxmax < 900:
					markpos = pdxmax-345
					break
			if markpos == -1:
				markpos = 330

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if ymin >= avgpdymin and ymax < avgpdymax and ((xwidth <= 46 and yheight <= 46) or xwidth*yheight < 1500):
					continue
				if ymin >= avgpdymin and ymax < avgpdymax and (xmin < 15 or xmax > 1265) and xwidth < 30:
					continue

				if ymin >= avgpdymax and  int(yheight*0.5333) < 55:
					continue

				if xmax > 1270 and xwidth <= 20 and yheight >= 5*xwidth and score < 0.9:
					continue


				if (ymin < avgpdymin-385) and ((xmin > markpos-425 and xmin < markpos-110 ) or (xmax > markpos-425 and xmax < markpos-130) or (xmid > markpos-425 and xmid < markpos-130)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
					
				if (ymin < avgpdymin-385) and (xmax > markpos+720):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmin < markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymax > avgpdymin - 230  and ymax < avgpdymin+50 and xmin > markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmax > markpos+590:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymin and ymid < avgpdymax+10:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
					if xwidth > 100 and score > 0.9:
						boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymax and ymid < avgpdymax+80 and xwidth*yheight > 60*60 and score > 0.9:
					boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

		if '78_MODULATORPADS' in imgpath:
			markpos = -1
			for pdbox in allpdboxlist:
				pdwidth = int(pdbox[3])-int(pdbox[1])
				pdxmin = int(pdbox[1])
				pdxmax = int(pdbox[3])
				if pdxmin < 100:
					markpos = pdxmax
					break
				elif pdxmax > 1230:
					markpos = pdxmin-930
					break
				elif pdwidth > 180 and pdxmax > 700:
					markpos = pdxmax-800
					break
				elif pdwidth < 130 and pdxmax > 400 and pdxmax < 800:
					markpos = pdxmax-345
					break
			if markpos == -1:
				markpos = 230

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if ymin >= avgpdymin and ymax < avgpdymax and ((xwidth <= 46 and yheight <= 46) or xwidth*yheight < 1500):
					continue
				if ymin >= avgpdymin and ymax < avgpdymax and (xmin < 15 or xmax > 1265) and xwidth < 30:
					continue

				if ymin >= avgpdymax and  int(yheight*0.5333) < 55:
					continue

				if xmax > 1230 and xwidth > 90 and yheight > 90 and score < 0.95:
					continue

				if (ymin < avgpdymin-385) and  xmin < markpos-130 :
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
					
				if (ymin < avgpdymin-385) and ((xmin > markpos+720 and xmin < markpos+1010 ) or (xmax > markpos+720 and xmax < markpos+1010) or (xmid > markpos+720 and xmid < markpos+1010)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmin < markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymax > avgpdymin - 230  and ymax < avgpdymin+50 and xmin > markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmax > markpos+590:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymin and ymid < avgpdymax+10:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
					if xwidth > 100 and score > 0.9:
						boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymax and ymid < avgpdymax+80 and xwidth*yheight > 60*60 and score > 0.9:
					boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


		if '79_MODULATORPADS' in imgpath:
			markpos = -1
			for pdbox in allpdboxlist:
				pdwidth = int(pdbox[3])-int(pdbox[1])
				pdxmin = int(pdbox[1])
				pdxmax = int(pdbox[3])
				if pdxmin < 50:
					markpos = pdxmax
					break
				elif pdxmax > 1230:
					markpos = pdxmin-930
					break
				elif pdwidth > 180 and pdxmax > 700 and pdxmax < 1100:
					markpos = pdxmax-800
					break
				elif pdwidth < 130 and pdxmax > 300 and pdxmax < 700:
					markpos = pdxmax-345
					break
			if markpos == -1:
				markpos = 140

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if ymin >= avgpdymin and ymax < avgpdymax and ((xwidth <= 46 and yheight <= 46) or xwidth*yheight < 1500):
					continue
				if ymin >= avgpdymin and ymax < avgpdymax and (xmin < 15 or xmax > 1265) and xwidth < 30:
					continue

				if ymin >= avgpdymax and  int(yheight*0.5333) < 55:
					continue

				if xmax > 1230 and xwidth > 90 and yheight > 90 and score < 0.96:
					continue
				if xmin < 15 and xwidth > 90 and yheight > 90 and score < 0.96:
					continue

				if (ymin < avgpdymin-385) and  xmin < markpos-130 :
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
					
				if (ymin < avgpdymin-385) and ((xmin > markpos+720 and xmin < markpos+1010 ) or (xmax > markpos+720 and xmax < markpos+1010) or (xmid > markpos+720 and xmid < markpos+1010)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmin < markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymax > avgpdymin - 230  and ymax < avgpdymin+50 and xmin > markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmax > markpos+590:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > avgpdymin and ymid < avgpdymax+10:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
					if xwidth > 100 and score > 0.9:
						boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue


				if ymid > avgpdymax and ymid < avgpdymax+80 and xwidth*yheight > 60*60 and score > 0.9:
					boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

		if '80_MODULATORPADS' in imgpath:
			markpos = -1
			for pdbox in allpdboxlist:
				pdwidth = int(pdbox[3])-int(pdbox[1])
				pdxmin = int(pdbox[1])
				pdxmax = int(pdbox[3])
				if pdxmin < 50:
					markpos = pdxmax+590
					break
				elif pdwidth > 180 and pdxmax > 600 and pdxmax < 1000:
					markpos = pdxmin
					break
				elif pdwidth > 180 and pdxmax > 1000:
					markpos = pdxmin-340
					break
				elif pdwidth < 130 and pdxmax > 200 and pdxmax < 600:
					markpos = pdxmax+245
					break
			if markpos == -1:
				markpos = 645

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if ymin >= avgpdymin and ymax < avgpdymax and ((xwidth <= 46 and yheight <= 46) or xwidth*yheight < 1500):
					continue
				if ymin >= avgpdymin and ymax < avgpdymax and (xmin < 15 or xmax > 1265) and xwidth < 30:
					continue

				if ymin >= avgpdymax and  int(yheight*0.5333) < 55:
					continue

				if xmax > 1230 and xwidth > 90 and yheight > 90 and score < 0.9:
					continue

				if xmin < 15 and xwidth > 90 and yheight > 90 and score < 0.9:
					continue

				if (ymin < avgpdymin-490) and  xmax > markpos+570:
					if xwidth >= 45 or yheight >= 45:
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
					
				if (ymin < avgpdymin-385) and ((xmin > markpos+130 and xmin < markpos+415 ) or (xmax > markpos+130 and xmax < markpos+415) or (xmid > markpos+130 and xmid < markpos+415)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin and xmax > markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymin > avgpdymin - 385  and ymin < avgpdymin  and xmin < markpos:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue


				if ymid > avgpdymin and ymid < avgpdymax+10:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
					if xwidth > 100 and score > 0.9:
						boxlist.append(objbox)
						tempngboxlist.append(objbox)
						continue
				
				if ymid > avgpdymax and ymid < avgpdymax+80 and xwidth*yheight > 60*60 and score > 0.9:
					boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

		if CheckUpDownTraceSize(boxlist,tempngboxlist,0,avgpdymin-370,20):
			return {},mdrate,[]

		if CheckUpDownTraceSize(boxlist,tempngboxlist,avgpdymin-370,1280,40):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	else:
		for objbox in allngboxlist:
			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2
			if '74_MODULATORPADS' in imgpath:
				if ymin < 550 and ((xmin > 180 and xmin < 470) or (xmax > 180 and xmax < 470) or (xmid > 180 and xmid < 470)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			elif '75_MODULATORPADS' in imgpath:
				if ymin < 550 and (((xmin > 90 and xmin < 390) or (xmax > 90 and xmax < 390) or (xmid > 90 and xmid < 390)) or xmax > 1220):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			elif '76_MODULATORPADS' in imgpath:
				if ymin < 550 and (xmin < 280 or xmax > 1130):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			elif '77_MODULATORPADS' in imgpath:
				if ymin < 550 and (xmin < 220 or xmax > 1030):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
			elif '78_MODULATORPADS' in imgpath:
				if ymin < 550 and (xmin < 100 or xmax > 940):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
			elif '79_MODULATORPADS' in imgpath:
				if ymin < 550 and (xmin < 20 or xmax > 850):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
			elif '80_MODULATORPADS' in imgpath:
				if ymin < 550 and ((xmin > 770 and xmin < 1070) or (xmax > 770 and xmax < 1070) or (xmid > 770 and xmid < 1070)):
					if (xwidth >= 35 and yheight >= 35) or (xwidth*yheight > 1200 and yheight >= 30):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist

		boxlist = []
		tempngboxlist = []

		for objbox in allngboxlist:
			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if '74_MODULATORPADS' in imgpath:
				if ymin < 400 and xmin < 25:
					if (xwidth >= 45 and yheight >= 45) :
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin > 530 and ymin < 1060 and xmin < 600:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > 700 and ymid < 1070:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			elif '75_MODULATORPADS' in imgpath:
				if ymin > 530 and ymin < 1060 and xmin < 520:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > 700 and ymid < 1070:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			elif '76_MODULATORPADS' in imgpath:
				if ymin > 530 and ymin < 1060 and xmin < 430:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymin > 530 and ymin < 1060 and xmax > 990:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > 700 and ymid < 1070:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			elif '77_MODULATORPADS' in imgpath:
				if ymin > 530 and ymin < 1060 and xmin < 360:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymin > 530 and ymin < 1060 and xmax > 890:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > 700 and ymid < 1070:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
			elif '78_MODULATORPADS' in imgpath:
				if ymin > 530 and ymin < 1060 and xmin < 250:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymin > 530 and ymin < 1060 and xmax > 800:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > 700 and ymid < 1070:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
			elif '79_MODULATORPADS' in imgpath:
				if ymin > 530 and ymin < 1060 and xmin < 170:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymin > 530 and ymin < 1060 and xmax > 700:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > 700 and ymid < 1070:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
			elif '80_MODULATORPADS' in imgpath:
				if ymax < 430 and xmax > 1210:
					if (xwidth >= 45 and yheight >= 45) :
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin > 530 and ymin < 1060 and xmax > 620  and xmin < 1210:
					if (xwidth >= 45 and yheight >= 45) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymax > 600 and xmin < 120:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

				if ymid > 700 and ymid < 1070:
					if (xwidth >= 50 and yheight >= 50) :
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

		if CheckUpDownTraceSize(boxlist,tempngboxlist,0,550,20):
			return {},mdrate,[]

		if CheckUpDownTraceSize(boxlist,tempngboxlist,550,1280,40):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]

	return {},mdrate,[]


def rawWH(box):
	ymin = int(box[0]*2048/1280)
	xmin = int(box[1]*2448/1280)
	ymax = int(box[2]*2048/1280)
	xmax = int(box[3]*2448/1280)
	xwidth = int(float(xmax-xmin)/120.0*40.0)
	yheight = int(float(ymax-ymin)/120.0*40.0)
	return xwidth,yheight


def SameBox(boxlist,distthold1=25):
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

			# print('dist1: '+str(dist1)+'..........')

			if dist1 > distthold1:
				return False
	return True


def CheckLeftPadTraceSize(boxlist,tempngboxlist,bound,defectsize=40):

	# print('boxlist len : '+str(len(boxlist))+'tempngboxlist len : '+str(len(tempngboxlist))+' bound '+str(bound)+'..................' )

	if len(tempngboxlist) > 0 and SameBox(tempngboxlist):
		if int(tempngboxlist[0][1]) > bound :
			w = -1
			h = -1
			for box in tempngboxlist:
				w1,h1 = rawWH(box)
				if w1 > w:
					w = w1
				if h1 > h:
					h = h1

			# print('tw: '+ str(w) + ' h: '+str(h)+'..................')
			if w <= defectsize and h <= defectsize: #and (w*w+h*h) < defectsize*defectsize:
				return True

	return False


def CheckRightPadTraceSize(boxlist,tempngboxlist,bound,defectsize=40):
	# print('boxlist len : '+str(len(boxlist))+' tempngboxlist len : '+str(len(tempngboxlist))+' bound '+str(bound)+'..................' )

	if len(tempngboxlist) > 0 and SameBox(tempngboxlist):
		if int(tempngboxlist[0][3]) < bound :
			w = -1
			h = -1
			for box in tempngboxlist:
				w1,h1 = rawWH(box)
				if w1 > w:
					w = w1
				if h1 > h:
					h = h1

			# print('tw: '+ str(w) + ' h: '+str(h)+'..................')
			if w <= defectsize and h <= defectsize: #and (w*w+h*h) < defectsize*defectsize:
				return True

	return False


def CheckUpDownTraceSize(boxlist,tempngboxlist,ubound,dbound,defectsize=40):

	if len(tempngboxlist) > 0 and SameBox(tempngboxlist):
		if (int(tempngboxlist[0][0]) > ubound  and int(tempngboxlist[0][0]) < dbound) or (int(tempngboxlist[0][2]) > ubound  and int(tempngboxlist[0][2]) < dbound):
			w = -1
			h = -1
			for box in tempngboxlist:
				w1,h1 = rawWH(box)
				if w1 > w:
					w = w1
				if h1 > h:
					h = h1

			# print('tw: '+ str(w) + ' h: '+str(h)+'..................')
			if w <= defectsize and h <= defectsize: #and (w*w+h*h) < defectsize*defectsize:
				return True
	return False


def CheckModulatorDefectSize(boxlist,tempngboxlist,defectsize=20):
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
		if w <= defectsize and h <= defectsize: #and (w*w+h*h) < defectsize*defectsize:
			return True
	return False

def RXPads(imgpath,allngboxlist,output_dict,mdrate,allpdboxlist,allngscorelist):
	sumcnt = 0
	sumleft = 0
	sumright = 0
	sumarea = 0
	avgleft = -1
	avgright = -1
	minarea = -1
	pdlow = 1280
	pdhigh = 0

	for pdbox in allpdboxlist:
		pdymin = int(pdbox[0])
		pdxmin = int(pdbox[1])
		pdymax = int(pdbox[2])
		pdxmax = int(pdbox[3])
		pdxwidth = pdxmax-pdxmin
		pdyheight = pdymax-pdymin

		if pdxmax > 500 and pdxwidth > 140 and pdyheight > 140:
			sumleft = sumleft + pdxmin
			sumright = sumright + pdxmax
			sumarea = sumarea+pdxwidth*pdyheight
			sumcnt = sumcnt + 1
			if pdymin < pdlow:
				pdlow = pdymin
			if pdymax > pdhigh:
				pdhigh = pdymax

	if sumcnt != 0:
		avgleft = int(sumleft/sumcnt)
		avgright = int(sumright/sumcnt)
		minarea = int(float(sumarea/sumcnt)*0.2)
		
		boxlist = []
		tempngboxlist = []
		
		if '1_RXPADS'  in imgpath:
			markpos = 875
			if pdlow < 750:
				markpos = pdlow+360
			if pdhigh > 800:
				markpos = pdhigh-155

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if (xmid > avgleft-105 and xmid <  avgleft -30) and (ymid > markpos-745 and ymid < markpos-680):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymax > markpos-300) and ((xmin > avgright+165 and xmin < avgright+210) or (xmax > avgright+165 and xmax < avgright+210) or (xmid > avgright+165 and xmid < avgright+210)) :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgright+40 and xmin < avgright+165) or (xmax >  avgright+40 and xmax < avgright+165) or (xmid >  avgright+40 and xmid < avgright+165))  and ((ymin > markpos-330 and ymin < markpos-250) or (ymax > markpos-330 and ymax < markpos-250) or (ymid > markpos-330 and ymid < markpos-250)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgright+40 and xmin < avgright+165) or (xmax >  avgright+40 and xmax < avgright+165) or (xmid >  avgright+40 and xmid < avgright+165))  and ((ymin > markpos+35 and ymin < markpos+105) or (ymax > markpos+35 and ymax < markpos+105) or (ymid > markpos+35 and ymid < markpos+105)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-60 and xmid < avgright+40) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,avgright):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '18_RXPADS'  in imgpath:
			markpos = 725
			if pdlow < 100:
				markpos = pdlow+725
			if pdhigh > 1100:
				markpos = pdhigh-515

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if ((xmin > avgright+120 and xmin < avgright+210) or (xmax >  avgright+120 and xmax < avgright+210) or (xmid >  avgright+120 and xmid < avgright+210)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmax > avgright+120) and ((ymin > markpos-135 and ymin < markpos+275) or (ymax > markpos-135 and ymax < markpos+275) or (ymid > markpos-135 and ymid < markpos+275)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if(ymax > markpos+260) and ((xmin > avgright+140 and xmin < avgright+380) or (xmax >  avgright+140 and xmax < avgright+380) or (xmid >  avgright+140 and xmid < avgright+380)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgright+40 and xmin < avgright+120) or (xmax >  avgright+40 and xmax < avgright+120) or (xmid >  avgright+40 and xmid < avgright+120)) and ((ymin > markpos-690 and ymin < markpos-610) or (ymax > markpos-690 and ymax < markpos-610) or (ymid > markpos-690 and ymid < markpos-610)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+120) or (xmax >  avgright+40 and xmax < avgright+120) or (xmid >  avgright+40 and xmid < avgright+120)) and ((ymin > markpos-325 and ymin < markpos-250) or (ymax > markpos-325 and ymax < markpos-250) or (ymid > markpos-325 and ymid < markpos-250)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+120) or (xmax >  avgright+40 and xmax < avgright+120) or (xmid >  avgright+40 and xmid < avgright+120)) and ((ymin > markpos+35 and ymin < markpos+105) or (ymax > markpos+35 and ymax < markpos+105) or (ymid > markpos+35 and ymid < markpos+105)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+120) or (xmax >  avgright+40 and xmax < avgright+120) or (xmid >  avgright+40 and xmid < avgright+120)) and ((ymin > markpos+400 and ymin < markpos+470) or (ymax > markpos+400 and ymax < markpos+470) or (ymid > markpos+400 and ymid < markpos+470)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmid > avgleft-60 and xmid < avgright+40) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,avgright):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '19_RXPADS'  in imgpath:
			markpos = 775
			if pdhigh > 1200:
				markpos = pdhigh-510			
			if pdlow < 130:
				markpos = pdlow+730

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if ((xmin > avgright+165 and xmin < avgright+385) or (xmax >  avgright+165 and xmax < avgright+385) or (xmid > avgright+165 and xmid < avgright+385)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymax > markpos+165) and (xmax > avgright+165):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmax > avgright+435) and ((ymin > markpos-510 and ymin < markpos) or (ymax > markpos-510 and ymax < markpos) or (ymid > markpos-510 and ymid < markpos)):
					if xmax >= 1275 and yheight > 75 and yheight < 90 and xwidth >25 and xwidth < 35:
						continue
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				
				if (xmid > avgleft-60 and xmid < avgright+40) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,avgright):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '36_RXPADS'  in imgpath:
			markpos = 620
			if pdlow < 180:
				markpos = pdlow+545
			if pdhigh > 1000:
				markpos = pdhigh-525

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmax > 1270 and xwidth <=22 and yheight >= 4.5*xwidth and score < 0.9:
					continue

				if (ymin < markpos+90) and (xmax > avgright+170):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymax > markpos+90) and (xmax > avgright+190):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgright+40 and xmin < avgright+170) or (xmax >  avgright+40 and xmax < avgright+170) or (xmid >  avgright+40 and xmid < avgright+170)) and ((ymin > markpos+35 and ymin < markpos+105) or (ymax > markpos+35 and ymax < markpos+105) or (ymid > markpos+35 and ymid < markpos+105)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+190) or (xmax >  avgright+40 and xmax < avgright+190) or (xmid >  avgright+40 and xmid < avgright+190)) and ((ymin > markpos+400 and ymin < markpos+480) or (ymax > markpos+400 and ymax < markpos+480) or (ymid > markpos+400 and ymid < markpos+480)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-60 and xmid < avgright+40) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-60 and xmid < avgright+40):
					if (xwidth >= 60 and yheight >= 60):
						boxlist.append(objbox)
					continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,avgright):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '37_RXPADS'  in imgpath:
			markpos = 840
			if pdlow < 300:
				markpos = pdlow+730
			if pdhigh > 800:
				markpos = pdhigh-160

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if (ymin < markpos-270) and (xmax > avgright+210):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymax > markpos-270) and (xmax > avgright+250):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgright+40 and xmin < avgright+250) or (xmax >  avgright+40 and xmax < avgright+250) or (xmid >  avgright+40 and xmid < avgright+250)) and ((ymin > markpos-690 and ymin < markpos-610) or (ymax > markpos-690 and ymax < markpos-610) or (ymid > markpos-690 and ymid < markpos-610)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				
				if ((xmin > avgright+40 and xmin < avgright+250) or (xmax >  avgright+40 and xmax < avgright+250) or (xmid >  avgright+40 and xmid < avgright+250)) and ((ymin > markpos-320 and ymin < markpos-250) or (ymax > markpos-320 and ymax < markpos-250) or (ymid > markpos-320 and ymid < markpos-250)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgright+40 and xmin < avgright+250) or (xmax >  avgright+40 and xmax < avgright+250) or (xmid >  avgright+40 and xmid < avgright+250)) and ((ymin > markpos+35 and ymin < markpos+105) or (ymax > markpos+35 and ymax < markpos+105) or (ymid > markpos+35 and ymid < markpos+105)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-60 and xmid < avgright+40) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,avgright):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '54_RXPADS'  in imgpath:
			markpos = 690
			if pdlow < 100:
				markpos = pdlow+720
			elif pdlow < 500:
				markpos = pdlow+360
			if pdhigh > 1000:
				markpos = pdhigh-515

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2


				if (ymin < markpos-280) and (xmax > avgright+285):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymax > markpos-270) and (xmax > avgright+310):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgright+40 and xmin < avgright+310) or (xmax >  avgright+40 and xmax < avgright+310) or (xmid >  avgright+40 and xmid < avgright+310)) and ((ymin > markpos-690 and ymin < markpos-610) or (ymax > markpos-690 and ymax < markpos-610) or (ymid > markpos-690 and ymid < markpos-610)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+310) or (xmax >  avgright+40 and xmax < avgright+310) or (xmid >  avgright+40 and xmid < avgright+310)) and ((ymin > markpos-325 and ymin < markpos-250) or (ymax > markpos-325 and ymax < markpos-250) or (ymid > markpos-325 and ymid < markpos-250)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+310) or (xmax >  avgright+40 and xmax < avgright+310) or (xmid >  avgright+40 and xmid < avgright+310)) and ((ymin > markpos+35 and ymin < markpos+105) or (ymax > markpos+35 and ymax < markpos+105) or (ymid > markpos+35 and ymid < markpos+105)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+310) or (xmax >  avgright+40 and xmax < avgright+310) or (xmid >  avgright+40 and xmid < avgright+310)) and ((ymin > markpos+400 and ymin < markpos+470) or (ymax > markpos+400 and ymax < markpos+470) or (ymid > markpos+400 and ymid < markpos+470)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmid > avgleft-60 and xmid < avgright+40) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,avgright):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '55_RXPADS'  in imgpath:
			markpos = 550
			if pdlow < 400:
				markpos = pdlow+370
			if pdhigh > 1150:
				markpos = pdhigh-705

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if (ymin < markpos+100) and (xmax > avgright+360):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymax > markpos+90) and (xmax > avgright+400):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgright+40 and xmin < avgright+400) or (xmax >  avgright+40 and xmax < avgright+400) or (xmid >  avgright+40 and xmid < avgright+400)) and ((ymin > markpos-325 and ymin < markpos-250) or (ymax > markpos-325 and ymax < markpos-250) or (ymid > markpos-325 and ymid < markpos-250)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+400) or (xmax >  avgright+40 and xmax < avgright+400) or (xmid >  avgright+40 and xmid < avgright+400)) and ((ymin > markpos+35 and ymin < markpos+105) or (ymax > markpos+35 and ymax < markpos+105) or (ymid > markpos+35 and ymid < markpos+105)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+400) or (xmax >  avgright+40 and xmax < avgright+400) or (xmid >  avgright+40 and xmid < avgright+400)) and ((ymin > markpos+400 and ymin < markpos+470) or (ymax > markpos+400 and ymax < markpos+470) or (ymid > markpos+400 and ymid < markpos+470)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-60 and xmid < avgright+40) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,avgright):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]

		elif '72_RXPADS'  in imgpath:
			markpos = 765
			
			if pdhigh > 1200:
				markpos = pdhigh-520
			elif  pdhigh > 1030:
				markpos = pdhigh-340
			if pdlow < 130:
				markpos = pdlow+730

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if (ymin < markpos-275) and (xmax > avgright+420):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymax > markpos-270) and (xmax > avgright+460):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgright+40 and xmin < avgright+460) or (xmax >  avgright+40 and xmax < avgright+460) or (xmid >  avgright+40 and xmid < avgright+460)) and ((ymin > markpos-690 and ymin < markpos-610) or (ymax > markpos-690 and ymax < markpos-610) or (ymid > markpos-690 and ymid < markpos-610)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+460) or (xmax >  avgright+40 and xmax < avgright+460) or (xmid >  avgright+40 and xmid < avgright+460)) and ((ymin > markpos-325 and ymin < markpos-250) or (ymax > markpos-325 and ymax < markpos-250) or (ymid > markpos-325 and ymid < markpos-250)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+460) or (xmax >  avgright+40 and xmax < avgright+460) or (xmid >  avgright+40 and xmid < avgright+460)) and ((ymin > markpos+35 and ymin < markpos+105) or (ymax > markpos+35 and ymax < markpos+105) or (ymid > markpos+35 and ymid < markpos+105)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if ((xmin > avgright+40 and xmin < avgright+460) or (xmax >  avgright+40 and xmax < avgright+460) or (xmid >  avgright+40 and xmid < avgright+460)) and ((ymin > markpos+400 and ymin < markpos+470) or (ymax > markpos+400 and ymax < markpos+470) or (ymid > markpos+400 and ymid < markpos+470)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-60 and xmid < avgright+40) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,avgright):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '73_RXPADS'  in imgpath:
			markpos = 610
			if pdlow < 30:
				markpos = pdlow+720
			elif pdlow < 130:
				markpos = pdlow+540
			if pdhigh > 880:
				markpos = pdhigh-340


			mdpadymax = -1
			for pdbox in allpdboxlist:
				pdymin1 = int(pdbox[0])
				pdxmin1 = int(pdbox[1])
				pdymax1 = int(pdbox[2])
				pdxmax1 = int(pdbox[3])
				if pdxmin1 > avgright+150:
					mdpadymax = pdymax1

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if mdpadymax != -1:
					if ymin >= mdpadymax:
						continue
					if xmid <= avgright:
						if ymid > mdpadymax -10:
							continue


				if (xmid > avgleft-105 and xmid <  avgleft -30) and (ymid > markpos+450 and ymid < markpos+580):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymin < markpos-200) and ((xmin > avgright+500 and xmin < avgright+560) or (xmax >  avgright+500 and xmax < avgright+560) or (xmid >  avgright+500 and xmid < avgright+560)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgright+40 and xmin < avgright+560) or (xmax >  avgright+40 and xmax < avgright+560) or (xmid >  avgright+40 and xmid < avgright+560)) and ((ymin > markpos-330 and ymin < markpos-260) or (ymax > markpos-330 and ymax < markpos-260) or (ymid > markpos-330 and ymid < markpos-260)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgright+330 and xmin < avgright+560) or (xmax >  avgright+330 and xmax < avgright+560) or (xmid >  avgright+330 and xmid < avgright+560)) and ((ymin > markpos-260 and ymin < markpos-190) or (ymax > markpos-260 and ymax < markpos-190) or (ymid > markpos-260 and ymid < markpos-190)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgright+40 and xmin < avgright+200) or (xmax >  avgright+40 and xmax < avgright+200) or (xmid >  avgright+40 and xmid < avgright+200)) and ((ymin > markpos-60 and ymin < markpos+100) or (ymax > markpos-60 and ymax < markpos+100) or (ymid > markpos-60 and ymid < markpos+100)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgright+200 and xmin < avgright+330) or (xmax >  avgright+200 and xmax < avgright+330) or (xmid >  avgright+200 and xmid < avgright+330)) and ((ymin > markpos-230 and ymin < markpos-60) or (ymax > markpos-230 and ymax < markpos-60) or (ymid > markpos-230 and ymid < markpos-60)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmax > avgright+110) and ((ymin > markpos+275 and ymin < markpos+440) or (ymax > markpos+275 and ymax < markpos+440) or (ymid > markpos+275 and ymid < markpos+440)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmax > avgright+300) and ((ymin > markpos+40 and ymin < markpos+275) or (ymax > markpos+40 and ymax < markpos+275) or (ymid > markpos+40 and ymid < markpos+275)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-60 and xmid < avgright+40) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,avgright):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
	else:
		if '1_RXPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if ymax > 570 and ((xmin > 860 and xmin < 920) or (xmax > 860 and xmax < 920) or (xmid > 860 and xmid < 920)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if xmid > 740 and xmid < 860 and ymid > 540 and ymid < 630 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 860 and ymid > 900 and ymid < 990:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 460 and xmid < 560 and ymid > 160 and ymid < 260 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymax > 330  and ((xmin > 520 and xmin <= 740) or (xmax > 520 and xmax < 740) or (xmid > 520 and xmid < 740)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,400):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '18_RXPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmid > 740 and xmid < 820 and ymid > 40 and ymid < 120:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 820 and ymid > 400 and ymid < 480:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 820 and ymid > 760 and ymid < 830 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 820 and ymid > 1120 and ymid < 1200:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmin > 820 and xmin < 910) or (xmax > 820 and xmax < 910) or (xmid > 820 and xmid < 910):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmax > 820 and  ((ymin > 590 and ymin < 1010) or (ymax > 590 and ymax < 1010) or (ymid > 590 and ymid < 1010)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ymax >= 980 and ((xmin > 820 and xmin < 1080) or (xmax > 820 and xmax < 1080) or (xmid > 820 and xmid < 1080 )):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmin > 520 and xmin <= 740) or (xmax > 520 and xmax < 740) or (xmid > 520 and xmid < 740):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,400):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '19_RXPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if (xmin > 870 and xmin < 1080) or (xmax > 870 and xmax < 1080) or (xmid > 870 and xmid < 1080):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymax >= 940 and  xmax > 870:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > 1140 and xmin < 1230) or (xmax > 1140 and xmax < 1230) or (xmid > 1140 and xmid < 1230))  and ((ymin > 260 and ymin < 770) or (ymax > 260 and ymax < 770) or (ymid > 260 and ymid < 770)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin > 520 and xmin <= 740) or (xmax > 520 and xmax < 740) or (xmid > 520 and xmid < 740):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,400):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '36_RXPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmid > 740 and xmid < 870 and ymid > 650 and ymid < 730 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 870 and ymid > 1010 and ymid < 1090:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmax > 870:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin > 520 and xmin <= 740) or (xmax > 520 and xmax < 740) or (xmid > 520 and xmid < 740):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,400):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '37_RXPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmid > 740 and xmid < 910 and ymid > 150 and ymid < 230 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 910 and ymid > 510 and ymid < 590:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 910 and ymid >870 and ymid < 950:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 910 and ymid > 1250:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmax > 910:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin > 520 and xmin <= 740) or (xmax > 520 and xmax < 740) or (xmid > 520 and xmid < 740):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,400):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '54_RXPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmid > 740 and xmid < 970 and ymid > 10 and ymid < 85 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 970 and ymid > 370 and ymid < 450:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 970 and ymid >730 and ymid < 810:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 970 and ymid > 1090 and ymid < 1170:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmax > 970:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin > 520 and xmin <= 740) or (xmax > 520 and xmax < 740) or (xmid > 520 and xmid < 740):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,400):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '55_RXPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmid > 740 and xmid < 1050 and ymid > 220 and ymid < 300 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 1050 and ymid > 580 and ymid < 670:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 1050 and ymid >950 and ymid < 1040:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmax > 1050:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin > 520 and xmin <= 740) or (xmax > 520 and xmax < 740) or (xmid > 520 and xmid < 740):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,400):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '72_RXPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmid > 740 and xmid < 1120 and ymid > 70 and ymid < 160 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 1120 and ymid > 440 and ymid < 520:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 1120 and ymid >800 and ymid < 890:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 740 and xmid < 1120 and ymid >1160 and ymid < 1250:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmax > 1120:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin > 520 and xmin <= 740) or (xmax > 520 and xmax < 740) or (xmid > 520 and xmid < 740):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,400):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '73_RXPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmid >= 780 and xmid <= 900 and ymid >= 540 and ymid <= 690:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid >= 900 and xmid <= 1020 and ymid >= 400 and ymid <= 540:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 460 and xmid < 560 and ymid > 1030 and ymid < 1130 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 740 and xmid < 1210 and ymid > 280 and ymid < 380 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 1020 and xmid < 1210 and ymid > 350 and ymid < 450:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmax > 1210 and ymin < 410:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmax > 810 and ymid > 670 and ymid < 1070:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymin < 960 and ((xmin > 520 and xmin <= 740) or (xmax > 520 and xmax < 740) or (xmid > 520 and xmid < 740)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckLeftPadTraceSize(boxlist,tempngboxlist,400):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]

	return {},mdrate,[]

def ControlPads(imgpath,allngboxlist,output_dict,mdrate,allpdboxlist,allngscorelist):

	sumcnt = 0
	sumleft = 0
	sumright = 0
	sumarea = 0
	avgleft = -1
	avgright = -1
	minarea = -1
	pdlow = 1280
	pdhigh = 0

	for pdbox in allpdboxlist:
		pdymin = int(pdbox[0])
		pdxmin = int(pdbox[1])
		pdymax = int(pdbox[2])
		pdxmax = int(pdbox[3])
		pdxwidth = pdxmax-pdxmin
		pdyheight = pdymax-pdymin

		if pdxmax > 500 and pdxwidth > 140 and pdyheight > 140:
			sumleft = sumleft + pdxmin
			sumright = sumright + pdxmax
			sumarea = sumarea+pdxwidth*pdyheight
			sumcnt = sumcnt + 1
			if pdymin < pdlow:
				pdlow = pdymin
			if pdymax > pdhigh:
				pdhigh = pdymax

	if sumcnt != 0:
		avgleft = int(sumleft/sumcnt)
		avgright = int(sumright/sumcnt)
		minarea = int(float(sumarea/sumcnt)*0.2)
		
		boxlist = []
		tempngboxlist = []
		
		if '9_CONTROLPADS'  in imgpath:
			markpos = 500
			if pdlow < 700:
				markpos = pdlow
			if pdhigh > 800:
				markpos = pdhigh - 520

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if (ymid > markpos-410 and ymid < markpos-240) and score > 0.985:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
						continue

				if (xmid > avgright+20 and xmid < avgright+100) and  (ymid > markpos-410 and ymid < markpos-240):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymax > markpos+60 and ((xmin > avgleft-215 and xmin < avgleft-180) or (xmax > avgleft-215 and xmax < avgleft-180) or (xmid > avgleft-215 and xmid < avgleft-180)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymax > markpos+420 and ((xmin > avgleft-215 and xmin < avgleft-155) or (xmax > avgleft-215 and xmax < avgleft-155) or (xmid > avgleft-215 and xmid < avgleft-155)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-215 and xmin < avgleft-40) or (xmax > avgleft-215 and xmax < avgleft-40) or (xmid > avgleft-215 and xmid < avgleft-40)) and ((ymin > markpos+40 and ymin < markpos+110) or (ymax > markpos+40 and ymax <  markpos+110) or (ymid > markpos+40 and ymid <  markpos+110)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-215 and xmin < avgleft-40) or (xmax > avgleft-215 and xmax < avgleft-40) or (xmid > avgleft-215 and xmid < avgleft-40)) and ((ymin >  markpos+400 and ymin < markpos+470) or (ymax > markpos+400 and ymax < markpos+470) or (ymid > markpos+400 and ymid < markpos+470)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-40 and xmid < avgright+60) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,avgleft):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]

		elif '10_CONTROLPADS'  in imgpath:
			markpos = 725
			if pdlow < 100:
				markpos = pdlow+720
			if pdhigh > 1000:
				markpos = pdhigh-520

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2


				if (xmin < avgleft-130) and ((ymin > markpos-135 and ymin < markpos+270) or (ymax > markpos-135 and ymax <  markpos+270) or (ymid > markpos-135 and ymid <  markpos+270)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymax > markpos+270) and ((xmin > avgleft-375 and xmin < avgleft-145) or (xmax > avgleft-375 and xmax < avgleft-145) or (xmid > avgleft-375 and xmid < avgleft-145)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymin < markpos-135) and ((xmin > avgleft-205 and xmin < avgleft-130) or (xmax >  avgleft-205 and xmax < avgleft-130) or (xmid >  avgleft-205 and xmid < avgleft-130)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgleft-145 and xmin < avgleft-40) or (xmax > avgleft-145 and xmax < avgleft-40) or (xmid > avgleft-145 and xmid < avgleft-40)) and ((ymin >  markpos-685 and ymin < markpos-610) or (ymax > markpos-685 and ymax < markpos-610) or (ymid > markpos-685 and ymid < markpos-610)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-145 and xmin < avgleft-40) or (xmax > avgleft-145 and xmax < avgleft-40) or (xmid > avgleft-145 and xmid < avgleft-40)) and ((ymin >  markpos-325 and ymin < markpos-250) or (ymax > markpos-325 and ymax < markpos-250) or (ymid > markpos-325  and ymid < markpos-250)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-145 and xmin < avgleft-40) or (xmax > avgleft-145 and xmax < avgleft-40) or (xmid > avgleft-145 and xmid < avgleft-40)) and ((ymin >  markpos+35 and ymin < markpos+110) or (ymax > markpos+35 and ymax < markpos+110) or (ymid > markpos+35 and ymid < markpos+110)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-145 and xmin < avgleft-40) or (xmax > avgleft-145 and xmax < avgleft-40) or (xmid > avgleft-145 and xmid < avgleft-40)) and ((ymin >  markpos+400 and ymin < markpos+475) or (ymax > markpos+400 and ymax < markpos+475) or (ymid > markpos+400 and ymid < markpos+475)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmid > avgleft-40 and xmid < avgright+60) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,avgleft):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]

		elif '27_CONTROLPADS'  in imgpath:
			markpos = 50
			if pdhigh > 1200:
				markpos = pdhigh-1235
			if pdlow < 150:
				markpos = pdlow

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2


				if ((xmin > avgleft-380 and xmin < avgleft-165) or (xmax > avgleft-380 and xmax < avgleft-165) or (xmid > avgleft-380 and xmid < avgleft-165)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmin < avgleft-430) and ((ymin > markpos+210 and ymin < markpos+715) or (ymax > markpos+210 and ymax <  markpos+715) or (ymid > markpos+210 and ymid <  markpos+715)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymax > markpos+880) and (xmin < avgleft-370):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-40 and xmid < avgright+60) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,avgleft):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]

		elif '28_CONTROLPADS'  in imgpath:
			markpos = 650
			if pdlow < 200:
				markpos = pdlow+550
			if pdhigh > 1000:
				markpos = pdhigh-520

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmin < avgleft-165 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin < avgleft-40) and ((ymin > markpos+40 and ymin < markpos+115) or (ymax > markpos+40 and ymax <  markpos+115) or (ymid > markpos+40 and ymid <  markpos+115)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin < avgleft-40) and ((ymin > markpos+400 and ymin < markpos+475) or (ymax > markpos+400 and ymax <  markpos+475) or (ymid > markpos+400 and ymid <  markpos+475)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-40 and xmid < avgright+60) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,avgleft):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]

		elif '45_CONTROLPADS'  in imgpath:
			markpos = 650
			if pdlow < 200:
				markpos = pdlow+550
			if pdhigh > 1000:
				markpos = pdhigh-520

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2


				if xmin < avgleft-165 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin < avgleft-40) and ((ymin > markpos+40 and ymin < markpos+115) or (ymax > markpos+40 and ymax <  markpos+115) or (ymid > markpos+40 and ymid <  markpos+115)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin < avgleft-40) and ((ymin > markpos+400 and ymin < markpos+475) or (ymax > markpos+400 and ymax <  markpos+475) or (ymid > markpos+400 and ymid <  markpos+475)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmid > avgleft-40 and xmid < avgright+60) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,avgleft):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '46_CONTROLPADS'  in imgpath:
			markpos = 730
			if pdlow < 100:
				markpos = pdlow+725
			if pdhigh > 1000:
				markpos = pdhigh-520

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2


				if (ymin < markpos-260) and (xmin < avgleft-270):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (ymax > markpos-260) and ( xmin < avgleft-310):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgleft-270 and xmin < avgleft-40) or (xmax >avgleft-270 and xmax < avgleft-40) or (xmid > avgleft-270 and xmid < avgleft-40)) and ((ymin >  markpos-685 and ymin < markpos-610) or (ymax > markpos-685 and ymax < markpos-610) or (ymid > markpos-685 and ymid < markpos-610)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-285 and xmin < avgleft-40) or (xmax > avgleft-285 and xmax < avgleft-40) or (xmid > avgleft-285 and xmid < avgleft-40)) and ((ymin >  markpos-325 and ymin < markpos-250) or (ymax > markpos-325 and ymax < markpos-250) or (ymid > markpos-325  and ymid < markpos-250)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-300 and xmin < avgleft-40) or (xmax > avgleft-300 and xmax < avgleft-40) or (xmid > avgleft-300 and xmid < avgleft-40)) and ((ymin >  markpos+35 and ymin < markpos+110) or (ymax > markpos+35 and ymax < markpos+110) or (ymid > markpos+35 and ymid < markpos+110)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-315 and xmin < avgleft-40) or (xmax > avgleft-315 and xmax < avgleft-40) or (xmid > avgleft-315 and xmid < avgleft-40)) and ((ymin >  markpos+400 and ymin < markpos+475) or (ymax > markpos+400 and ymax < markpos+475) or (ymid > markpos+400 and ymid < markpos+475)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmid > avgleft-40 and xmid < avgright+60) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,avgleft):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '63_CONTROLPADS'  in imgpath:
			markpos = 570
			if pdhigh > 1100:
				markpos = pdhigh-700
			if pdlow < 400:
				markpos = pdlow+360

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if (ymin < markpos+70) and (xmin < avgleft-360):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (ymax > markpos+70) and ( xmin < avgleft-400):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgleft-360 and xmin < avgleft-40) or (xmax >avgleft-360 and xmax < avgleft-40) or (xmid > avgleft-360 and xmid < avgleft-40)) and ((ymin >  markpos-330 and ymin < markpos-230) or (ymax > markpos-330 and ymax < markpos-230) or (ymid > markpos-250 and ymid < markpos-230)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-380 and xmin < avgleft-40) or (xmax > avgleft-380  and xmax < avgleft-40) or (xmid > avgleft-380  and xmid < avgleft-40)) and ((ymin >  markpos+35 and ymin < markpos+135) or (ymax > markpos+35 and ymax < markpos+135) or (ymid > markpos+35 and ymid < markpos+135)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-400 and xmin < avgleft-40) or (xmax > avgleft-400 and xmax < avgleft-40) or (xmid > avgleft-400 and xmid < avgleft-40)) and ((ymin >  markpos+400 and ymin < markpos+500) or (ymax > markpos+400 and ymax < markpos+500) or (ymid > markpos+400 and ymid < markpos+500)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmid > avgleft-40 and xmid < avgright+60) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,avgleft):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '64_CONTROLPADS'  in imgpath:
			markpos = 770
			if pdhigh > 1200:
				markpos = pdhigh-505
			if pdlow < 120:
				markpos = pdlow+730

			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2


				if (ymin < markpos-280) and (xmin < avgleft-430):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (ymax > markpos-280) and ( xmin < avgleft-470):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgleft-470 and xmin < avgleft-40) or (xmax >avgleft-470 and xmax < avgleft-40) or (xmid > avgleft-470 and xmid < avgleft-40)) and ((ymin >  markpos-700 and ymin < markpos-610) or (ymax > markpos-700 and ymax < markpos-610) or (ymid > markpos-700 and ymid < markpos-610)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-470 and xmin < avgleft-40) or (avgleft-470  and xmax < avgleft-40) or (xmid > avgleft-470  and xmid < avgleft-40)) and ((ymin >  markpos-340 and ymin < markpos-250) or (ymax > markpos-340 and ymax < markpos-250) or (ymid > markpos-340 and ymid < markpos-250)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-470 and xmin < avgleft-40) or (avgleft-470 and xmax < avgleft-40) or (xmid > avgleft-470 and xmid < avgleft-40)) and ((ymin >  markpos+30 and ymin < markpos+120) or (ymax > markpos+30 and ymax < markpos+120) or (ymid > markpos+30 and ymid < markpos+120)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-470 and xmin < avgleft-40) or (avgleft-470 and xmax < avgleft-40) or (xmid > avgleft-470 and xmid < avgleft-40)) and ((ymin >  markpos+390 and ymin < markpos+480) or (ymax > markpos+390 and ymax < markpos+480) or (ymid > markpos+390 and ymid < markpos+480)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmid > avgleft-40 and xmid < avgright+60) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,avgleft):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
			
		elif '81_CONTROLPADS'  in imgpath:
			markpos = 620
			if pdlow < 160:
				markpos = pdlow+545
			if pdhigh > 850:
				markpos = pdhigh-325


			mdpadymax = -1
			for pdbox in allpdboxlist:
				pdymin1 = int(pdbox[0])
				pdxmin1 = int(pdbox[1])
				pdymax1 = int(pdbox[2])
				pdxmax1 = int(pdbox[3])
				if pdxmax1 < avgleft-150:
					mdpadymax = pdymax1


			idx = 0
			for objbox in allngboxlist:
				score = allngscorelist[idx]
				idx = idx + 1

				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if mdpadymax != -1:
					if ymin >= mdpadymax:
						continue
					if xmid >= avgleft:
						if ymid > mdpadymax -10:
							continue

				if (xmid > avgright+20 and xmid < avgright+100) and  (ymid > markpos+400 and ymid < markpos+570):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (ymin < markpos-200) and (xmin < avgleft-520):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgleft-520 and xmin < avgleft-40) or (xmax >avgleft-520 and xmax < avgleft-40) or (xmid > avgleft-520 and xmid < avgleft-40)) and ((ymin >  markpos-330 and ymin < markpos-240) or (ymax > markpos-330 and ymax < markpos-240) or (ymid > markpos-330 and ymid < markpos-240)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > avgleft-520 and xmin < avgleft-340) or (xmax >avgleft-520 and xmax < avgleft-340) or (xmid > avgleft-520 and xmid < avgleft-340)) and ((ymin >  markpos-250 and ymin < markpos-160) or (ymax > markpos-250 and ymax < markpos-160) or (ymid > markpos-250 and ymid < markpos-160)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-340 and xmin < avgleft-210) or (avgleft-340 and xmax < avgleft-210) or (xmid > avgleft-340 and xmid < avgleft-210)) and ((ymin >  markpos-230 and ymin < markpos-70) or (ymax > markpos-230 and ymax < markpos-70) or (ymid > markpos-230 and ymid < markpos-70)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-210 and xmin < avgleft-40) or (avgleft-210 and xmax < avgleft-40) or (xmid > avgleft-210 and xmid < avgleft-40)) and ((ymin >  markpos-70 and ymin < markpos+90) or (ymax > markpos-70 and ymax < markpos+90) or (ymid > markpos-70 and ymid < markpos+90)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > avgleft-460 and xmin < avgleft-120) or (avgleft-460 and xmax < avgleft-120) or (xmid > avgleft-460 and xmid < avgleft-120)) and ((ymin >  markpos+270 and ymin < markpos+440) or (ymax > markpos+270 and ymax < markpos+440) or (ymid > markpos+270 and ymid < markpos+440)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if (xmin < avgleft-300) and ((ymin >  markpos+90 and ymin < markpos+270) or (ymax > markpos+90 and ymax < markpos+270) or (ymid > markpos+90 and ymid < markpos+270)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if (xmid > avgleft-40 and xmid < avgright+60) and (xwidth*yheight > minarea or (xwidth*yheight > minarea/2 and score > 0.95)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,avgleft):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
	else:
		if '9_CONTROLPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmid > 660 and xmid < 770 and ymid > 160 and ymid < 260 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymax > 570 and ((xmin > 320 and xmin < 390) or (xmax > 320 and xmax < 390) or (xmid > 320 and xmid < 390)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 320 and xmid < 510 and ymid > 540 and ymid < 630 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 320 and xmid < 510 and ymid > 920 and ymid < 1000:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymax > 330  and ((xmin > 500 and xmin <= 720) or (xmax > 500 and xmax < 720) or (xmid > 500 and xmid < 720)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue
			
			if CheckRightPadTraceSize(boxlist,tempngboxlist,1000):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '10_CONTROLPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmid > 420 and xmid < 510 and ymid > 40 and ymid < 120 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 420 and xmid < 510 and ymid > 410 and ymid < 490 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 420 and xmid < 510 and ymid > 770 and ymid < 850 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmid > 420 and xmid < 510 and ymid > 1140 and ymid < 1220 :
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > 330 and xmin < 420) or (xmax > 330 and xmax < 420) or (xmid > 330 and xmid < 420)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmin < 340 and ((ymin > 590 and ymin < 1020) or (ymax > 590 and ymax < 1020) or (ymid > 590 and ymid < 1020)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ymax > 1010 and ((xmin > 160 and xmin < 400) or (xmax > 160 and xmax < 400) or (xmid > 160 and xmid < 400)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > 490 and xmin < 720) or (xmax > 490 and xmax < 720) or (xmid > 490 and xmid < 720)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,1000):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]

		elif '27_CONTROLPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmin < 110 and ((ymin > 250 and ymin < 770) or (ymax > 250 and ymax < 770) or (ymid > 250 and ymid < 770)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > 150 and xmin < 370) or (xmax > 150 and xmax < 370) or (xmid > 150 and xmid < 370)):
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmin < 160 and ymax > 930:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > 490 and xmin < 720) or (xmax > 490 and xmax < 720) or (xmid > 490 and xmid < 720)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,1000):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '28_CONTROLPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmin < 370:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 420 and xmid < 510 and ymid > 640 and ymid < 740:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 420 and xmid < 510 and ymid > 1020 and ymid < 1130:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > 490 and xmin < 720) or (xmax > 490 and xmax < 720) or (xmid > 490 and xmid < 720)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,1000):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '45_CONTROLPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmin < 320:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 320 and xmid < 510 and ymid > 140 and ymid < 240:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 320 and xmid < 510 and ymid > 500 and ymid < 600:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 320 and xmid < 510 and ymid > 860 and ymid < 960:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 320 and xmid < 510 and ymax > 1250:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > 490 and xmin < 720) or (xmax > 490 and xmax < 720) or (xmid > 490 and xmid < 720)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,1000):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '46_CONTROLPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmin < 190 and ymax > 1130:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 210 and ymin < 1140:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 230 and ymin < 780:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 250 and ymin < 410:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 510 and ymin < 70:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if xmid > 190 and xmid < 510 and ymid > 1090 and ymid < 1190:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 210 and xmid < 510 and ymid > 730 and ymid < 830:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 230 and xmid < 510 and ymid > 370 and ymid < 470:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > 490 and xmin < 720) or (xmax > 490 and xmax < 720) or (xmid > 490 and xmid < 720)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,1000):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
		elif '63_CONTROLPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmin < 130 and ymax > 970:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 150 and ymin < 980:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 170 and ymin < 620:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 190 and ymin < 260:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if xmid > 120 and xmid < 510 and ymid > 930 and ymid < 1020:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 140 and xmid < 510 and ymid > 560 and ymid < 660:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 170 and xmid < 510 and ymid > 200 and ymid < 300:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > 490 and xmin < 720) or (xmax > 490 and xmax < 720) or (xmid > 490 and xmid < 720)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,1000):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]

		elif '64_CONTROLPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmin < 30 and ymax > 1210:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 45 and ymin < 1220:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 65 and ymin < 860:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 90 and ymin < 490:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if xmin < 110 and ymin < 130:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if xmid > 30 and xmid < 510 and ymid > 1150 and ymid < 1250:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 60 and xmid < 510 and ymid > 790 and ymid < 890:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 80 and xmid < 510 and ymid > 430 and ymid < 530:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 110 and xmid < 510 and ymid > 60 and ymid < 160:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


				if ((xmin > 490 and xmin < 720) or (xmax > 490 and xmax < 720) or (xmid > 490 and xmid < 720)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,1000):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]

		elif '81_CONTROLPADS'  in imgpath:
			boxlist = []
			tempngboxlist = []

			for objbox in allngboxlist:
				ymin = int(objbox[0])
				xmin = int(objbox[1])
				ymax = int(objbox[2])
				xmax = int(objbox[3])
				xwidth = xmax-xmin
				yheight = ymax-ymin
				xmid = (xmin+xmax)/2
				ymid = (ymin+ymax)/2

				if xmin < 35 and ymin < 400:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 30 and xmid < 510 and ymid > 270 and ymid < 370:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmin < 210 and ymid > 360 and ymid < 460:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid >= 210 and xmid <= 360 and ymid >= 400 and ymid <= 575:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid >= 360 and xmid <= 500 and ymid >= 575 and ymid <= 730:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmin < 430 and ymid > 680 and ymid < 1110:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if xmid > 680 and xmid < 780 and ymid > 1030 and ymid < 1130:
					if (xwidth >= 45 and yheight >= 45):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

				if ((xmin > 490 and xmin < 720) or (xmax > 490 and xmax < 720) or (xmid > 490 and xmid < 720)):
					if (xwidth >= 45 and yheight >= 45):
						if CheckPadS(objbox,allpdboxlist):
							boxlist.append(objbox)
					if CheckPadS(objbox,allpdboxlist):
						tempngboxlist.append(objbox)
						continue

			if CheckRightPadTraceSize(boxlist,tempngboxlist,1000):
				return {},mdrate,[]

			if len(boxlist) > 0:
				return output_dict,mdrate,boxlist
			elif len(tempngboxlist) > 0:
				if checkboxdist(tempngboxlist,distthold1=45):
					return output_dict,mdrate,boxlist
			else:
				return {},mdrate,[]
	return {},mdrate,[]

def WaveGuide(imgpath,allngboxlist,output_dict,mdrate,allngscorelist,filterversion):
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
		xwidth = xmax-xmin
		yheight = ymax-ymin
		xmid = (xmin+xmax)/2
		ymid = (ymin+ymax)/2

		# if ((ymin > 80 and ymin < 240) or (ymax > 80 and ymax < 240) or (ymid > 80 and ymid < 240)):
		# 	if filterversion == 'V1':
		# 		if (xwidth >= 2.2*yheight and yheight <= 28 and xwidth >= 44 and score < 0.9) and ymax < 230:
		# 			continue
		# 		if (xwidth >= 2.2*yheight and yheight <= 19 and xwidth >= 38 and score < 0.9) and ymax < 230:
		# 			continue


		# 	if filterversion == 'V2':
		# 		if xwidth*yheight > 6000 and score < 0.51:
		# 			continue
		# 		if (xwidth >= 2.2*yheight and yheight <= 28 and xwidth >= 44 and score < 0.92) and ymax < 230:
		# 			continue
		# 		if (xwidth >= 2.2*yheight and yheight <= 19 and xwidth >= 38 and score < 0.92) and ymax < 230:
		# 			continue


		if ((ymin > 80 and ymin < 280) or (ymax > 80 and ymax < 280) or (ymid > 80 and ymid < 280)):
			if '82_WAVEGUIDE' in imgpath:
				if (xmid > 80 and xmid < 450) or (xmid > 850 and xmid < 1200):
					boxlist.append(objbox)
			elif '83_WAVEGUIDE' in imgpath or '84_WAVEGUIDE' in imgpath or '85_WAVEGUIDE' in imgpath or '86_WAVEGUIDE' in imgpath:
				if (xmid > 430 and xmid < 870):
					boxlist.append(objbox)
			else:
				boxlist.append(objbox)

	if len(boxlist) > 0:
		return output_dict,mdrate,boxlist
	else:
		return {},mdrate,[]

	# if '82_WAVEGUIDE' in imgpath:
	# elif '83_WAVEGUIDE' in imgpath:
	# elif '84_WAVEGUIDE' in imgpath:
	# elif '85_WAVEGUIDE' in imgpath:
	# elif '86_WAVEGUIDE' in imgpath:
	return {},mdrate,[]

def Heater(imgpath,allngboxlist,allwdboxlist,output_dict,mdrate,allngscorelist):

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

	if avgwdymin != -1:
		idx = 0
		for objbox in allngboxlist:
			score = allngscorelist[idx]
			idx = idx + 1

			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			for wdbox in allwdboxlist:
				wdymin = int(wdbox[0])
				wdxmin = int(wdbox[1])
				wdymax = int(wdbox[2])
				wdxmax = int(wdbox[3])
				if((ymin > wdymin and ymin < wdymax) or (ymax > wdymin and ymax < wdymax) or (ymid > wdymin and ymid < wdymax))  and ((xmin > wdxmin and xmin < wdxmax) or (xmax > wdxmin and xmax < wdxmax) or (xmid > wdxmin and xmid < wdxmax)):
					if (xwidth >= 70 or yheight >= 70):
						boxlist.append(objbox)
					elif (xwidth >= 45 or yheight >= 45) and score > 0.985:
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					break

			if leftwd < 550:
				vleftwd = leftwd+700
				vrightwd = leftwd+860
				if((ymin > avgwdymin and ymin < avgwdymax) or (ymax > avgwdymin and ymax < avgwdymax) or (ymid > avgwdymin and ymid < avgwdymax)) and ((xmin > vleftwd and xmin < vrightwd) or (xmax > vleftwd and xmax < vrightwd) or (xmid > vleftwd and xmid < vrightwd)):
					if (xwidth >= 70 or yheight >= 70):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

			elif leftwd > 550:
				vleftwd = leftwd-700
				vrightwd = leftwd - 540
				if((ymin > avgwdymin and ymin < avgwdymax) or (ymax > avgwdymin and ymax < avgwdymax) or (ymid > avgwdymin and ymid < avgwdymax)) and ((xmin > vleftwd and xmin < vrightwd) or (xmax > vleftwd and xmax < vrightwd) or (xmid > vleftwd and xmid < vrightwd)):
					if (xwidth >= 70 or yheight >= 70):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue


			if ymax > avgwdymax+160:
				if (xwidth >= 70 or yheight >= 70):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if leftwd < 550:
				vleftwd = leftwd
				if (ymax > (avgwdymax + 60)) and ((xmin > vleftwd+60 and xmin < vleftwd+800) or (xmax > vleftwd+60 and xmax < vleftwd+800) or (xmid > vleftwd+60 and xmid < vleftwd+800)):
					if (xwidth >= 70 or yheight >= 70):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if (ymin < (avgwdymin -40)) and ((xmin > vleftwd and xmin < vleftwd+110) or (xmax > vleftwd and xmax < vleftwd+110) or (xmid > vleftwd and xmid < vleftwd+110)):
					if (xwidth >= 70 or yheight >= 70):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
			elif leftwd > 550:
				vleftwd = leftwd-700
				if (ymax > (avgwdymax + 60)) and ((xmin > vleftwd+60 and xmin < vleftwd+800) or (xmax > vleftwd+60 and xmax < vleftwd+800) or (xmid > vleftwd+60 and xmid < vleftwd+800)):
					if (xwidth >= 70 or yheight >= 70):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue
				if (ymin < (avgwdymin -40)) and ((xmin > vleftwd and xmin < vleftwd+110) or (xmax > vleftwd and xmax < vleftwd+110) or (xmid > vleftwd and xmid < vleftwd+110)):
					if (xwidth >= 70 or yheight >= 70):
						boxlist.append(objbox)
					tempngboxlist.append(objbox)
					continue

		if CheckUpDownTraceSize(boxlist,tempngboxlist,avgwdymax+70,1280,80):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist,65,200):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]
		return {},mdrate,[]
	else:
		for objbox in allngboxlist:
			ymin = int(objbox[0])
			xmin = int(objbox[1])
			ymax = int(objbox[2])
			xmax = int(objbox[3])
			xwidth = xmax-xmin
			yheight = ymax-ymin
			xmid = (xmin+xmax)/2
			ymid = (ymin+ymax)/2

			if ymax > 920:
				if (xwidth >= 70 or yheight >= 70):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if ymin < 300 and xmid > 110 and xmid < 300:
				if (xwidth >= 70 or yheight >= 70):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if ymin < 300 and xmid > 900 and xmid < 1050:
				if (xwidth >= 70 or yheight >= 70):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if xmid > 100 and xmid < 370 and ymid > 230 and ymid < 930 :
				if (xwidth >= 70 or yheight >= 70):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

			if xmid > 790 and xmid < 1070 and ymid > 230 and ymid < 930 :
				if (xwidth >= 70 or yheight >= 70):
					boxlist.append(objbox)
				tempngboxlist.append(objbox)
				continue

		if CheckUpDownTraceSize(boxlist,tempngboxlist,0,1280,80):
			return {},mdrate,[]

		if len(boxlist) > 0:
			return output_dict,mdrate,boxlist
		elif len(tempngboxlist) > 0:
			if checkboxdist(tempngboxlist,65,200):
				return output_dict,mdrate,boxlist
		else:
			return {},mdrate,[]
		# if '87_HEATER_' in imgpath:
		# elif '88_HEATER_' in imgpath:
		# elif '89_HEATER_' in imgpath:
		# elif '90_HEATER_' in imgpath:
		# elif '91_HEATER_' in imgpath:
		# elif '92_HEATER_' in imgpath:
		# elif '93_HEATER_' in imgpath:
		# elif '94_HEATER_' in imgpath:
		return {},mdrate,[]

def GetOutPutDict(param,allscorelist1,output_dict1,allscorelist2,output_dict2,allscorelist3,output_dict3,allngboxlist,allpdboxlist,allwdboxlist,allngscorelist,filterversion):
	
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

	if '2_AROUNDWAVEGUIDE' in imgpath or '3_AROUNDWAVEGUIDE' in imgpath or '4_AROUNDWAVEGUIDE' in imgpath or '5_AROUNDWAVEGUIDE' in imgpath or '6_AROUNDWAVEGUIDE' in imgpath or '7_AROUNDWAVEGUIDE' in imgpath  or '8_AROUNDWAVEGUIDE' in imgpath:
		return AroundWaveGuide(imgpath,allngboxlist,output_dict,mdrate)
	elif '11_METALTRACE' in imgpath or '12_METALTRACE' in imgpath or '13_METALTRACE' in imgpath or '14_METALTRACE' in imgpath or '15_METALTRACE' in imgpath or '16_METALTRACE' in imgpath  or '17_METALTRACE' in imgpath:
		return MetalTrace1(allngboxlist,output_dict,mdrate)
	elif '21_METALTRACE' in imgpath or '22_METALTRACE' in imgpath or '23_METALTRACE' in imgpath or '24_METALTRACE' in imgpath or '25_METALTRACE' in imgpath:
		return MetalTrace2(imgpath,allngboxlist,allwdboxlist,output_dict,mdrate)
	elif '20_METALTRACE' in imgpath or '26_METALTRACE' in imgpath:
		return MetalTrace3(imgpath,allngboxlist,allwdboxlist,output_dict,mdrate)
	elif '_MODULATOR_'  in imgpath:
		return Modulator(imgpath,allngboxlist,output_dict,mdrate,allngscorelist)
	elif '74_MODULATORPADS_' in imgpath or '75_MODULATORPADS_' in imgpath or '76_MODULATORPADS_' in imgpath or '77_MODULATORPADS_' in imgpath or '78_MODULATORPADS_' in imgpath or '79_MODULATORPADS_' in imgpath or '80_MODULATORPADS_' in imgpath:
		return ModulatorPads(imgpath,allngboxlist,output_dict,mdrate,allpdboxlist,allngscorelist)
	elif '_RXPADS_' in imgpath:
		return RXPads(imgpath,allngboxlist,output_dict,mdrate,allpdboxlist,allngscorelist)
	elif '_CONTROLPADS_' in imgpath:
		return ControlPads(imgpath,allngboxlist,output_dict,mdrate,allpdboxlist,allngscorelist)
	elif '_WAVEGUIDE_' in imgpath:
		return WaveGuide(imgpath,allngboxlist,output_dict,mdrate,allngscorelist,filterversion)
	elif '_HEATER_' in imgpath:
		return Heater(imgpath,allngboxlist,allwdboxlist,output_dict,mdrate,allngscorelist)

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
		# if MatchBox(objbox,output_dict3,param):
		# 	return output_dict3
	return output_dict


def ImagePrepare(param):
	newimgpath = ''
	cimg = None
	cimgx = None

	try:
		upperpath = param.rawpath.upper()
		gamma = 0.9

		if '_WAVEGUIDE_' in upperpath:
			param.score = 0.4
			gamma = 0.7

		fileexist = os.path.exists(param.rawpath)
		if fileexist:
			cimgx = cv2.imread(param.rawpath,cv2.IMREAD_COLOR)
			cimg = cv2.resize(cimgx,(WIDTH,HIGH))
			cimg = CLAHE(cimg,2.1,16)
			cimg = gamma_correction_lab(cimg,gamma)

			newimgpath = SaveTobeImg(cimg,param)
			if newimgpath == '':
				return
		else:
			if param.tobepath != '':
				tobexist =  os.path.exists(param.tobepath)
				if tobexist:
					newimgpath = param.tobepath
					cimg = cv2.imread(param.tobepath,cv2.IMREAD_COLOR)
					cimgx = cv2.resize(cimg,(2448,2048))
				else:
					NoRAWImg(param)
					return
			else:
				NoRAWImg(param)
				return

		param.newimgpath = newimgpath
		param.cimgx = cimgx
		rgbimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2RGB)
		param.img_tensor = tf.convert_to_tensor(rgbimg,dtype=tf.float32)
	except:
		NoRAWImg(param)
		traceback.print_exc()
	return


def run_model(param):
	try:
		if param.newimgpath == '':
			return None

		upperpath = param.rawpath.upper()
		if '_WAVEGUIDE_' in upperpath:
			param.score = 0.4
		
		# img = tf.io.read_file(newimgpath)
		# img_tensor = tf.io.decode_image(img, channels=3)

		img_tensor = param.img_tensor

		# input_image_size = (HIGH, WIDTH)
		# img_tensor = tf.image.resize(img_tensor,input_image_size)

		img_tensor = tf.expand_dims(img_tensor, axis=0)
		img_tensor = tf.cast(img_tensor, dtype = tf.uint8)
		
		output_dict1 = param.model_fn(img_tensor)
		output_dict2 = param.model_fn2(img_tensor)
		# output_dict3 = param.model_fn3(img_tensor)
		output_dict3=[]

		del param.img_tensor
		param.img_tensor = None

		allpdboxlist80 = []
		allpdboxlist90= []
		allwdboxlist80 = []
		allwdboxlist90 = []
		allngboxlist = []
		allngscorelist = []

		allscorelist1 = GetObjectList(output_dict1,param.score,allngboxlist,allpdboxlist80,allpdboxlist90,allwdboxlist80,allwdboxlist90,allngscorelist)
		allscorelist2 = GetObjectList(output_dict2,param.score,allngboxlist,allpdboxlist80,allpdboxlist90,allwdboxlist80,allwdboxlist90,allngscorelist)
		# allscorelist3 = GetObjectList(output_dict3,param.score,allngboxlist,allpdboxlist80,allpdboxlist90,allwdboxlist80,allwdboxlist90,allngscorelist)
		allscorelist3=[]
		
		allwdboxlist = allwdboxlist80
		if len(allwdboxlist90) > 0:
			allwdboxlist = allwdboxlist90
		
		allpdboxlist = allpdboxlist80
		if len(allpdboxlist90) > 0:
			allpdboxlist = allpdboxlist90

		output_dict,mdrate,boxlist = GetOutPutDict(param,allscorelist1,output_dict1,allscorelist2,output_dict2,allscorelist3,output_dict3,allngboxlist,allpdboxlist,allwdboxlist,allngscorelist,'V1')
		if 'detection_scores' in output_dict and len(boxlist) > 0:
			output_dict = GetMatchOutputDict(boxlist,output_dict,output_dict1,output_dict2,output_dict3,param)

		maxscore = 0
		NGsubfix = '.jpg'
		AOIRest = 'PASS'
		if 'detection_scores' in output_dict:
			for i in range(100):
				if float(output_dict['detection_scores'][0][i]) >= param.score:
					clsidx = int(output_dict['detection_classes'][0][i]) - 1
					tscore = round(100.0*float(output_dict['detection_scores'][0][i]),2)

					if clsidx == 0:
						AOIRest = 'FAIL'
						NGsubfix = '_NG.jpg'
						if int(tscore) > maxscore:
							maxscore = int(tscore)

					color = param.colors[clsidx%30]
					bbox = output_dict['detection_boxes'][0][i]
					drawtangle2(bbox,clsidx,param.cimgx,color,tscore)

		anlyzedpath = SaveAnalyzedImg(param.cimgx,param,NGsubfix)
		del param.cimgx
		param.cimgx = None

		return AOIRESTITEM(param.aoikey,param.pj,param.wafer,param.cellpos,param.newimgpath,anlyzedpath,AOIRest,maxscore,mdrate)
	except:
		exception_message = sys.exc_info()[1]
		print(param.rawpath)
		print(str(exception_message))
		if '!ssize.empty()' in str(exception_message):
			NoRAWImg(param)
		traceback.print_exc()
		return None



def StoreAOIResult(AOIRESTList):
	try:
		myclient = pymongo.MongoClient(DBCONNECTSTR)
		mydb = myclient["NPITrace"]
		aoicol = mydb["PICDIEAOI"]
		for item in AOIRESTList:
			query = {'_id':ObjectId(item.aoikey)}
			setval = {'$set':{'Analyzed':1,'AOIResult':item.aoirest,'ToBePath':item.tobepath,'AnalyzedPath':item.analyzepath,'AnalyzeTime':item.analyzedtime,'MaxScore':item.maxscore,'MDRate':item.MDRate}}
			aoicol.update_many(query,setval)
	except:
		print('a database except happend2.......')
		exception_message = sys.exc_info()[1]
		print(str(exception_message))
		time.sleep(5)

def NoRAWImg(param):
	try:
		myclient = pymongo.MongoClient(DBCONNECTSTR)
		mydb = myclient["NPITrace"]
		aoicol = mydb["PICDIEAOI"]
		query = {'_id':ObjectId(param.aoikey)}
		setval = {'$set':{'Analyzed':2}}
		aoicol.update_many(query,setval)
	except:
		print('a database except happend2.......')
		exception_message = sys.exc_info()[1]
		print(str(exception_message))
		time.sleep(5)


def SplitAOIList(lst,chunk_size=50):
	return [lst[i:i+chunk_size] for i in range(0,len(lst),chunk_size)]

def ImagePrepareParallel(chunck):
	executor = ThreadPoolExecutor(4)
	all_tasks = [executor.submit(ImagePrepare,param) for param in chunck]
	wait(all_tasks,return_when=ALL_COMPLETED)


def MainLoop():

	parser = argparse.ArgumentParser()
	parser.add_argument('--gpuid', type=str, help='gpu device id', required=True)
	parser.add_argument('--modnum', type=int, help='mod number', required=True)
	parser.add_argument('--runid', type=int, help='run id', required=True)
	parser.add_argument('--modnum2', type=int, help='mod again num',default=0)
	parser.add_argument('--runid2', type=int, help='mod again run id', default=0)
	args = parser.parse_args()

	print('python PICDIE_AOI.py  --gpuid  '+args.gpuid+'  --modnum  '+str(args.modnum)+'  --runid  '+str(args.runid)+'  --modnum2  '+str(args.modnum2)+'  --runid2  '+str(args.runid2))

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
	gpus = tf.config.list_physical_devices('GPU')
	tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6.5*1024)])
	logical_gpus = tf.config.list_logical_devices('GPU')
	print(logical_gpus)

	modnum = args.modnum
	myrunid = args.runid

	modnum2 = args.modnum2
	myrunid2 = args.runid2

	with tf.device('/device:GPU:0'):
		sleepbase = 5
		sleepidx = 1

		while(True):
			now = datetime.now()
			nowtime = now.strftime("%Y-%m-%d %H:%M:%S")

			print('python PICDIE_AOI.py  --gpuid  '+args.gpuid+'  --modnum  '+str(args.modnum)+'  --runid  '+str(args.runid)+'  --modnum2  '+str(args.modnum2)+'  --runid2  '+str(args.runid2))
			print('loading to be analyzed SWD AOI data................'+nowtime)

			AOIItemList = GetAOIItems(modnum,myrunid,modnum2,myrunid2)

			paramlistlen = len(AOIItemList)
			print('the  SWD AOI list count is '+str(paramlistlen))

			if paramlistlen == 0 and sleepidx < 12:
				sleepidx = sleepidx + 1
			if paramlistlen > 0:
				sleepidx = 1

			prevtime = datetime.now()
			solved = 0
			AOIRESTList = []
			AOIChuncks = SplitAOIList(AOIItemList)
			for chunck in AOIChuncks:
				ImagePrepareParallel(chunck)
				for param in chunck:
					paramdate = param.uptime.strftime("%Y-%m-%d %H:%M:%S")
					rest = run_model(param)
					if rest != None:
						AOIRESTList.append(rest)

					if len(AOIRESTList) == 10:
						StoreAOIResult(AOIRESTList)
						AOIRESTList = []
						gc.collect()

						solved = solved+10
						if solved%1000 == 0:
							print('python PICDIE_AOI.py  --gpuid  '+args.gpuid+'  --modnum  '+str(args.modnum)+'  --runid  '+str(args.runid)+'  --modnum2  '+str(args.modnum2)+'  --runid2  '+str(args.runid2))
							now = datetime.now()
							nowtime = now.strftime("%Y-%m-%d %H:%M:%S")
							print('to be analyzed AOI data is '+str(paramlistlen-solved)+'....param date '+paramdate+'.....TS..'+nowtime)

							timespan = (now-prevtime).seconds
							if timespan > 2700:
								print("WARNING: the network seems have problem, it spend "+str(timespan)+" seconds to complete 1000 tasks")
							prevtime = datetime.now()

			if len(AOIRESTList) > 0:
				StoreAOIResult(AOIRESTList)
				
			time.sleep(sleepbase*sleepidx)


if __name__ == "__main__":
	MainLoop()
