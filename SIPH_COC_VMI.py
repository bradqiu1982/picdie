import argparse
import os
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
from datetime import datetime


wblock = threading.Lock()
HIGH = 1280
WIDTH = 1280
model_cache = dict()
OCRscore = 0.6
CLAscore = 0.2
classlistdict = {}
pkgcla = {}

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
	retcolors.append(colors[7])
	retcolors.append(colors[25])
	retcolors.append(colors[13])
	retcolors.append(colors[23])
	retcolors.append(colors[28])
	for cc in colors:
		retcolors.append(cc)
	return retcolors

# ['BHP1','Furu70','Furu100','Sumi']
def  getSIPHOCRModel(coctype):
	if coctype not in model_cache:
		export_dir = './OCR/SIPH_COC/'+coctype+'/exported_model'
		imported = tf.saved_model.load(export_dir)
		model_fn = imported.signatures['serving_default']
		model_cache[coctype] = model_fn
		return model_fn
	else:
		return model_cache[coctype]

def  getSIPHCLAModel():
	SIPHCLAMODEL='SIPHCLAMODEL'
	if SIPHCLAMODEL not in model_cache:
		export_dir = './CLA/SIPHCOC/exported_model_v5_996'
		imported = tf.saved_model.load(export_dir)
		model_fn = imported.signatures['serving_default']
		model_cache[SIPHCLAMODEL] = model_fn
		return model_fn
	else:
		return model_cache[SIPHCLAMODEL]

class SIPHCOCITEM:
	def __init__(self,aoikey,tobepath,sn,pkgid,lock,colors,OCRscore,ocr_model_fn,CLAscore,cla_model_fn):
		self.aoikey = aoikey
		self.tobepath = tobepath.upper()
		self.sn = sn
		self.pkgid = pkgid
		self.lock = lock
		self.colors = colors
		self.OCRscore = OCRscore
		self.ocr_model_fn = ocr_model_fn
		self.CLAscore = CLAscore
		self.cla_model_fn = cla_model_fn
		self.coctypestr = ''

class OCRITEM:
	def __init__(self,claidx,ymin,ymax,xmin,xmax):
		self.claidx = claidx
		self.ymin = ymin
		self.ymax = ymax
		self.xmin = xmin
		self.xmax = xmax

def getRunID(filepath,modnum):
    try:
        fps = filepath.split("\\")
        if(len(fps) < 2):
            fps = filepath.split("/")
        ids = fps[len(fps)-1].split("_")
        idx = int(ids[0])%modnum
    except:
        print('exception file path in runid:'+filepath)
        return 0
    return idx


def convertRawimg(rawpath):
    fps = rawpath.split("\\")
    if(len(fps) < 2):
        fps = filepath.split("/")
    fname = fps[len(fps)-1]

    now = datetime.now()
    nowtime = now.strftime("%Y-%m-%d")
    tobepath = "\\\\cnwx-cifs\\Datacom_Test_Data03\\WUXI_AI02\\AOI\\SIPHCOC_OCR\\"+nowtime+"\\"

    if not os.path.exists(tobepath):
        os.makedirs(tobepath)
    analyzepath = tobepath.replace('SIPHCOC_OCR','SIPHCOC_OCR_ANALYZED')
    if not os.path.exists(analyzepath):
        os.makedirs(analyzepath)

    tobefile = tobepath+fname

    img = cv2.imread(rawpath,cv2.IMREAD_COLOR);
    img = cv2.resize(img,(1280,848));
    array_created = np.full((1280, 1280, 3),0, dtype = np.uint8)
    array_created[0:848,0:1280] = img
    cv2.imwrite(tobefile,array_created,[cv2.IMWRITE_JPEG_QUALITY,100])

    return tobefile.upper()



def GetAOIItems(modnum,myrunid):
	colors = random_colors(30)
	ocr_model_fn = getSIPHOCRModel('Furu70')
	cla_model_fn = getSIPHCLAModel()

	AOIItemList = []

	try:
		sql = "select aoikey,tobepath,sn,pkgid,rawpath from [WAT].[dbo].[AOI]  with(nolock)  where (project = 'SIPHCOC_OCR' or project = 'SIPH_OCR') and parsed = 0 and updatetime > '2026-05-22 00:00:01' order by pkgid,updatetime asc"
		with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wux-engsys01.chn.ii-vi.net', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
			cursor = conn.cursor()
			cursor.execute(sql)
			rows = cursor.fetchall() 
			for row in rows:
				runid = getRunID(str(row[1]),modnum)
				if runid != myrunid:
					continue
				
				tobe = str(row[1]).lower().replace('\\wux-fs','\\datacom-fs')
				raw = str(row[4]).lower().replace('\\wux-fs','\\datacom-fs')

				fileexist = os.path.exists(tobe)
				if not fileexist:
					rawexist = os.path.exists(raw)
					if not rawexist:
						continue
					else:
						tobe = convertRawimg(raw)
						toexist = os.path.exists(tobe)
						if not toexist:
							continue

				item = SIPHCOCITEM(str(row[0]),tobe,str(row[2]),str(row[3]),wblock,colors,OCRscore,ocr_model_fn,CLAscore,cla_model_fn)
				AOIItemList.append(item)
			cursor.close()
	except:
		exception_message = sys.exc_info()[1]
		print(str(exception_message))

	return AOIItemList


def StoreAOIResult(aoikey,analyzepath,aoiraw,aoirest,coctypestr,subimg,OCRWarning):
	now = datetime.now()
	analyzedtime = now.strftime("%Y-%m-%d %H:%M:%S")
	try:
		sql = 'update [WAT].[dbo].[AOI] set analyzepath=\''+analyzepath+'\',aoirest=\''+aoirest+'\',parsed = 1,appv_1=\''+coctypestr+'\',subimg=\''+subimg+'\',appv_2=\''+OCRWarning+'\',analyzedtime=\''+analyzedtime+'\' where aoikey=\''+aoikey+'\'';
		with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wux-engsys01.chn.ii-vi.net', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
			cursor = conn.cursor()
			cursor.execute(sql)
	except:
		exception_message = sys.exc_info()[1]
		print(str(exception_message))

	# ,aoiraw=\''+aoiraw+'\'
	# with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wux-engsys01.chn.ii-vi.net', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
	# 	cursor = conn.cursor()
	# 	sql = 'delete from [WAT].[dbo].[AOIAddionalInfo]  where aoikey=\''+aoikey+'\'';
	# 	cursor.execute(sql)
	# 	sql = 'insert into [WAT].[dbo].[AOIAddionalInfo](aoikey,aoiraw) values('+'\''+aoikey+'\''+','+'\''+aoiraw+'\')';
	# 	cursor.execute(sql)


def drawtangle2(box,cls,cimg,color,OCRWarning):
	ymin = int(box[0])
	xmin = int(box[1])
	ymax = int(box[2])
	xmax = int(box[3])

	# cx = int((xmin+xmax)/2)
	# cy = int((ymin+ymax)/2)

	cv2.rectangle(cimg,(xmin,ymin),(xmax,ymax),(0,255,0),2)
	
	mask =  np.ones((ymax-ymin,xmax-xmin), dtype="uint8")
	rnnmask = np.zeros((HIGH,WIDTH), dtype="uint8")
	rnnmask[ymin:ymax,xmin:xmax] = mask

	alpha=0.4
	for c in range(3):
		cimg[:, :, c] = np.where(rnnmask >= 1,cimg[:, :, c] *(1 - alpha) + alpha * color[c] * 255,cimg[:, :, c])


	fxmin = xmin+5
	fymin = ymin-25
	if fymin < 0:
		fymin = 0
	cv2.putText(cimg,str(int(cls)),(fxmin,fymin),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
	if OCRWarning != '':
		cv2.putText(cimg,OCRWarning,(20,800),cv2.FONT_HERSHEY_SIMPLEX,1,(25,140,255),2)



def run_model_again(param):
	cimg = cv2.imread(param.tobepath,cv2.IMREAD_COLOR)
	cimg = cv2.addWeighted(cimg, 1, np.zeros(cimg.shape, cimg.dtype), 0, -30)
	# cimg = cv2.medianBlur(cimg, 5)
	cv2.imwrite(param.tobepath.replace('.JPG','_1.JPG'),cimg)

	# input_image_size = (HIGH, WIDTH)
	img = tf.io.read_file(param.tobepath.replace('.JPG','_1.JPG'))
	img_tensor = tf.io.decode_image(img, channels=3)
	# img_tensor = tf.image.resize(img_tensor,input_image_size)
	img_tensor = tf.expand_dims(img_tensor, axis=0)
	img_tensor = tf.cast(img_tensor, dtype = tf.uint8)

	classlist = classlistdict[param.coctypestr]
	OCRWarning = ''
	ocrlist = []

	output_dict = {}
	param.lock.acquire()
	try:
		output_dict = param.ocr_model_fn(img_tensor)
	except:
		print('run into exception1.......')
	finally:
		param.lock.release()
	# print(output_dict)


	cimg = cv2.imread(param.tobepath,cv2.IMREAD_COLOR)
	# cimg = cv2.resize(cimg,(WIDTH,HIGH))
	leftedge,rightedge = GetOCREdge(output_dict,param)

	for i in range(100):
		detect_score = float(output_dict['detection_scores'][0][i])
		if detect_score >= param.OCRscore:
			clsidx = int(output_dict['detection_classes'][0][i]) - 1
			color = param.colors[clsidx%30]
			drawtangle2(output_dict['detection_boxes'][0][i],clsidx,cimg,color,OCRWarning)

			box = output_dict['detection_boxes'][0][i]
			ymin = int(box[0])
			xmin = int(box[1])
			ymax = int(box[2])
			xmax = int(box[3])

			if clsidx in classlist:
				if xmin > leftedge and xmax < rightedge:
					if (xmax-xmin) > 80:
						continue

					if detect_score < 0.9:
						OCRWarning = 'OCRWarning'

					ocritem = OCRITEM(clsidx,ymin,ymax,xmin,xmax)
					dup = False
					for vm in ocrlist:
						vmxmid = int((vm.xmin+vm.xmax)/2)
						if vmxmid > xmin and vmxmid < xmax:
							dup = True
							break
					if not dup:
						ocrlist.append(ocritem)


	output_dict_obj = {}
	output_dict_obj['detection_scores'] = output_dict['detection_scores'].numpy().tolist()
	output_dict_obj['detection_classes'] = output_dict['detection_classes'].numpy().tolist()
	output_dict_obj['detection_boxes'] = output_dict['detection_boxes'].numpy().tolist()
	# output_dict_obj['detection_masks'] = output_dict['detection_masks'].numpy().tolist()
	aoiraw = json.dumps(output_dict_obj)

	analyzepath = param.tobepath.replace('SIPHCOC_OCR','SIPHCOC_OCR_ANALYZED').replace('SIPH_OOCR','SIPH_OOCR_ANALYZED')
	cv2.imwrite(analyzepath,cimg)

	if len(ocrlist) > 3:
		ocrrest,subleft,subright,subtop,subbot = getOCRRest(param,ocrlist,classlist)
		subimg64str = getSubImg(param,cimg,subleft,subright,subtop,subbot)
		StoreAOIResult(param.aoikey,analyzepath,aoiraw,ocrrest,param.coctypestr,subimg64str,OCRWarning)
	else:
		StoreAOIResult(param.aoikey,'','','1234',param.coctypestr,'','OCRWarning')


def GetOCREdge(output_dict,param):
	leftedge = -1
	rightedge = -1
	leftedge2 = -1
	rightedge2 = -1
	furumarkwidth = -1

	for i in range(100):
		detect_score = float(output_dict['detection_scores'][0][i])
		if detect_score >= param.OCRscore:
			clsidx = int(output_dict['detection_classes'][0][i]) - 1
			
			box = output_dict['detection_boxes'][0][i]
			ymin = int(box[0])
			xmin = int(box[1])
			ymax = int(box[2])
			xmax = int(box[3])

			if param.coctypestr == 'Furu70':
				if clsidx == 10:
					leftedge = xmax+100
					rightedge2 = xmax+400
					objwidth = xmax-xmin
					furumarkwidth = xmax-xmin
					if objwidth > 220:
						leftedge = xmax+150
						rightedge2 = xmax+550

				if clsidx == 11:
					plugleft = 800
					if furumarkwidth > 220:
						plugleft = 1000
					if xmin > plugleft:
						rightedge = xmin - 90
						leftedge2 = xmin - 500

	if rightedge2 != -1 and rightedge ==-1:
		rightedge=rightedge2
	if leftedge2 != -1 and leftedge ==-1:
		leftedge=leftedge2

	if leftedge == -1:
		leftedge = 10
	if rightedge == -1:
		rightedge = 1270

	return leftedge,rightedge


def run_model(param):
	if not os.path.exists(param.tobepath):
			return None
	input_image_size = (HIGH, WIDTH)
	img = tf.io.read_file(param.tobepath)
	img_tensor = tf.io.decode_image(img, channels=3)
	# img_tensor = tf.image.resize(img_tensor,input_image_size)
	img_tensor = tf.expand_dims(img_tensor, axis=0)
	img_tensor = tf.cast(img_tensor, dtype = tf.uint8)

	classlist = classlistdict[param.coctypestr]
	OCRWarning = ''
	ocrlist = []

	output_dict = {}
	param.lock.acquire()
	try:
		output_dict = param.ocr_model_fn(img_tensor)
	except:
		print('run into exception1.......')
	finally:
		param.lock.release()
	# print(output_dict)

	CUDAFail = False
	try:
		cimg = cv2.imread(param.tobepath,cv2.IMREAD_COLOR)
		# cimg = cv2.resize(cimg,(WIDTH,HIGH))
		leftedge,rightedge = GetOCREdge(output_dict,param)

		for i in range(100):
			detect_score = float(output_dict['detection_scores'][0][i])
			if detect_score >= param.OCRscore:
				clsidx = int(output_dict['detection_classes'][0][i]) - 1
				color = param.colors[clsidx%30]
				drawtangle2(output_dict['detection_boxes'][0][i],clsidx,cimg,color,OCRWarning)

				box = output_dict['detection_boxes'][0][i]
				ymin = int(box[0])
				xmin = int(box[1])
				ymax = int(box[2])
				xmax = int(box[3])

				if clsidx in classlist:
					if xmin > leftedge and xmax < rightedge:
						if (xmax-xmin) > 80:
							continue

						if detect_score < 0.9:
							OCRWarning = 'OCRWarning'

						ocritem = OCRITEM(clsidx,ymin,ymax,xmin,xmax)
						dup = False
						for vm in ocrlist:
							vmxmid = int((vm.xmin+vm.xmax)/2)
							if vmxmid > xmin and vmxmid < xmax:
								dup = True
								break
						if not dup:
							ocrlist.append(ocritem)


		output_dict_obj = {}
		output_dict_obj['detection_scores'] = output_dict['detection_scores'].numpy().tolist()
		output_dict_obj['detection_classes'] = output_dict['detection_classes'].numpy().tolist()
		output_dict_obj['detection_boxes'] = output_dict['detection_boxes'].numpy().tolist()
		# output_dict_obj['detection_masks'] = output_dict['detection_masks'].numpy().tolist()
		aoiraw = json.dumps(output_dict_obj)

		analyzepath = param.tobepath.replace('SIPHCOC_OCR','SIPHCOC_OCR_ANALYZED').replace('SIPH_OOCR','SIPH_OOCR_ANALYZED')
		cv2.imwrite(analyzepath,cimg)

		if len(ocrlist) > 4:
			ocrrest,subleft,subright,subtop,subbot = getOCRRest(param,ocrlist,classlist)
			subimg64str = getSubImg(param,cimg,subleft,subright,subtop,subbot)
			StoreAOIResult(param.aoikey,analyzepath,aoiraw,ocrrest,param.coctypestr,subimg64str,OCRWarning)
		else:
			run_model_again(param)

	except:
		print('run into exception2.......')
		exception_message = sys.exc_info()[1]
		print(str(exception_message))
		print(param.tobepath)
		CUDAFail = True
		time.sleep(0.01)

	# if CUDAFail:
	# 	print('try again 1............')
	# 	CUDAFail = False
	# 	try:
	# 		run_model_again(param)
	# 	except:
	# 		print('run into exception3.......')
	# 		CUDAFail = True

	# if CUDAFail:
	# 	print('try again 2............')
	# 	CUDAFail = False
	# 	try:
	# 		run_model_again(param)
	# 	except:
	# 		print('run into exception4.......')
	# 		CUDAFail = True

def getOCRRest(param,ocrlist,classlist):
	ocrrest = ''
	subleft = 1281
	subright = 0
	subtop = 1281
	subbot = 0

	charxlist = []
	charxdict = {}
	ymindict = {}
	ymaxdict = {}

	for ocritem in ocrlist:
		xord = int((ocritem.xmin+ocritem.xmax)/2)
		charxlist.append(xord)
		charxdict[xord]= ocritem.claidx
		ymindict[xord] = int(ocritem.ymin)
		ymaxdict[xord] = int(ocritem.ymax)

	charxlist.sort()

	if param.coctypestr == 'Furu70' and len(charxlist) > 5:
		startidx = len(charxlist)-5
		charxlist=charxlist[startidx:]

	for idx in charxlist:

		if idx < subleft:
			subleft = idx
		if idx > subright:
			subright = idx
		if ymindict[idx] < subtop:
			subtop = ymindict[idx]
		if ymaxdict[idx] > subbot:
			subbot = ymaxdict[idx]

		ocrrest += str(classlist[charxdict[idx]])
		if param.coctypestr == 'Furu70' and len(ocrrest) == 5:
			return ocrrest,subleft-30,subright+30,subtop-3,subbot+3

	if subleft != 1281 and subright != 0:
		return ocrrest,subleft-30,subright+30,subtop-3,subbot+3
	else:
		return ocrrest,-1,-1,-1,-1

def getSubImg(param,cimg,subleft,subright,subtop,subbot):

	if subleft-6 < 0:
		subleft = 6

	if subright -50 < 0:
		subright = 50
	if subright+5 >1280:
		subright = 1270

	subimg = cimg[subtop:subbot,subleft:subright]
	res, im_png = cv2.imencode('.png', subimg)
	subimg64 = base64.b64encode(im_png.tobytes())
	subimg64str = subimg64.decode()
	return subimg64str

def getSIPHCOCTYPE(param):
	ref_dict = {}
	ref_dict[0] = 'BHP1'
	ref_dict[1] = 'Furu70'
	ref_dict[2] = 'Furu100'
	ref_dict[3] = 'Sumi'
	ref_dict[4] = 'NONE'
	# ref_dict[3] = 'Shijia100'
	# ref_dict[4] = 'yuanjie100'
	# ref_dict[5] = 'NONE'

	if param.pkgid in pkgcla:
		param.coctypestr = pkgcla[param.pkgid]
		param.ocr_model_fn = getSIPHOCRModel(param.coctypestr)
	else:
		input_image_size = (640, 640)
		img = tf.io.read_file(param.tobepath)
		img_tensor = tf.io.decode_image(img, channels=3)
		img_tensor = tf.image.resize(img_tensor,input_image_size)
		img_tensor = tf.expand_dims(img_tensor, axis=0)
		img_tensor = tf.cast(img_tensor, dtype = tf.uint8)

		output_dict = param.cla_model_fn(img_tensor)
		index = (tf.argmax(output_dict['logits'], axis=1)[0]).numpy()
		prob = (output_dict['probs'][0][index]).numpy()

		if prob >= param.CLAscore and index != 4:
			param.coctypestr = ref_dict[index]
			pkgcla[param.pkgid] = param.coctypestr
			param.ocr_model_fn = getSIPHOCRModel(param.coctypestr)
		else:
			param.coctypestr = ref_dict[4]


# ['BHP1','Furu70','Furu100','Sumi']
def MainLoop():

	parser = argparse.ArgumentParser()
	parser.add_argument('--gpuid', type=str, help='gpu device id', required=True)
	parser.add_argument('--modnum', type=int, help='mod number', required=True)
	parser.add_argument('--runid', type=int, help='run id', required=True)
	args = parser.parse_args()

	print('python SIPH_COC_VMI.py  --gpuid  '+args.gpuid+'  --modnum  '+str(args.modnum)+'  --runid  '+str(args.runid))

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
	gpus = tf.config.list_physical_devices('GPU')
	tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4.5*1024)])
	logical_gpus = tf.config.list_logical_devices('GPU')
	print(logical_gpus)

	modnum = args.modnum	
	myrunid = args.runid

	with tf.device('/device:GPU:0'):
		classlistdict['BHP1']=[0,1,2,3,4,5,6,7,8,9]
		classlistdict['Furu70']=[0,1,2,3,4,5,6,7,8,9]
		classlistdict['Furu100']=[0,1,2,3,4,5,6,7,8,9]
		classlistdict['Sumi']=[0,1,2,3,4,5,6,7,8,9]
		# classlistdict['Shijia100']=[0,1,2,3,4,5,6,7,8,9]
		# classlistdict['yuanjie100']=[0,1,2,3,4,5,6,7,8,9]

		print('loading ocr models................')
		getSIPHOCRModel('BHP1')
		getSIPHOCRModel('Furu70')
		getSIPHOCRModel('Furu100')
		getSIPHOCRModel('Sumi')
		# getSIPHOCRModel('Shijia100')
		# getSIPHOCRModel('yuanjie100')

		print('loading cla model................')
		getSIPHCLAModel()

		sleepbase = 5
		sleepidx = 1

		while(True):
			print('python SIPH_COC_VMI.py  --gpuid  '+args.gpuid+'  --modnum  '+str(args.modnum)+'  --runid  '+str(args.runid))
			
			now = datetime.now()
			nowtime = now.strftime("%Y-%m-%d %H:%M:%S")

			AOIItemList = GetAOIItems(modnum,myrunid)
			print('loading to be analyzed data count.....'+str(len(AOIItemList))+'...........'+nowtime)


			# AOIItemList100 = []
			# idx = 0
			# for item in AOIItemList:
			# 	AOIItemList100.append(item)
			# 	if idx > 12:
			# 		break
			# 	idx = idx + 1

			for item in AOIItemList:
				getSIPHCOCTYPE(item)

			AvailableItemList = []
			for item in AOIItemList:
				if item.coctypestr != 'NONE':
					AvailableItemList.append(item)

			paramlistlen = len(AvailableItemList)
			print('classfied files count .........'+str(paramlistlen)+'............')

			if paramlistlen == 0 and sleepidx < 7:
				sleepidx = sleepidx + 1
			if paramlistlen > 0:
				sleepidx = 1

			# if paramlistlen < 5:
			for param in AvailableItemList:
				run_model(param)
			# else:
			# 	executor = ThreadPoolExecutor(4)
			# 	all_tasks = [executor.submit(run_model,param) for param in AvailableItemList]
			# 	wait(all_tasks,return_when=ALL_COMPLETED)

			time.sleep(sleepbase*sleepidx)

if __name__ == "__main__":
	MainLoop()
