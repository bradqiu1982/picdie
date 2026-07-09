import pathlib
import os
from PIL import Image
import numpy as np
import cv2
import copy
import uuid
import random
import tensorflow as tf

from official.vision.data import tfrecord_lib

# from object_detection.utils import dataset_util

# from object_detection.utils import ops as utils_ops
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import math

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

import hashlib

import base64
import io
import pyodbc


class imagedata():
    """docstring for ClassName"""
    def __init__(self):
        self.height = 0
        self.width = 0
        self.filename = ''
        self.xmin = []
        self.xmax = []
        self.ymin = []
        self.ymax = []
        self.clsname = []
        self.clslabel = []
        self.iscrowd = []
        self.area = []


def convertimag_resize(imagepath,width,high):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    idx = 110
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        print(fn)
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            img = cv2.imread(fn,cv2.IMREAD_COLOR)
            img = cv2.resize(img,(width,high))
            nfile = '\\\\wux-engsys01\\PlanningForCast\\condaenv\\VISION\\zurich\\newimg\\'+str(idx)+'.jpg'
            cv2.imwrite(nfile,img,[cv2.IMWRITE_JPEG_QUALITY,100])
            idx = idx + 1




def convertimag2gray(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    idx = 110
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        print(fn)
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            img = cv2.imread(fn,cv2.IMREAD_COLOR)
            grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            cv2.imwrite(fn,grayimg,[cv2.IMWRITE_JPEG_QUALITY,100])



def NewImgResizeByWidth(fn,width):
    cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
    hg,wd,ch = cimg.shape
    nhg = int((float(width)/float(wd))*float(hg))
    cimg = cv2.resize(cimg,(width,nhg))
    array_created = np.full((width, width, 3),0, dtype = np.uint8)
    array_created[0:nhg,0:width] = cimg
    return array_created

def convertimag_bywidth(imagepath,despath,width):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        print(fn)
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            filename = os.path.basename(fn).upper().replace('.JPEG','.JPG').replace('.PNG','.JPG').replace('.BMP','.JPG')
            newimg = NewImgResizeByWidth(fn,width)
            cv2.imwrite(despath+'/'+ filename,newimg,[cv2.IMWRITE_JPEG_QUALITY,100])


def xml_to_csv0(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (int(root.find('size')[1].text),
                     int(root.find('size')[0].text),
                     int(member[4][0].text),
                     int(member[4][2].text),
                     int(member[4][1].text),
                     int(member[4][3].text),
                     root.find('filename').text,
                     member[0].text
                     )
            xml_list.append(value)
    column_name = [ 'height','width', 'xmin', 'xmax', 'ymin', 'ymax', 'filename', 'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def XML2CSV(xmlpath,csvfile):
    # image_path = os.path.join(os.getcwd(), 'PADVRF')
    xml_df = xml_to_csv0(xmlpath)
    xml_df.to_csv(csvfile, index=None)
    print('Successfully converted xml to csv.')


def create_tf_example(ih,iw,xmin,xmax,ymin,ymax,fpath,clatx,claid):

    # TODO(user): Populate the following variables from your example.
    height = ih # Image height
    width = iw # Image width
    filename = b'' # Filename of the image. Empty if image is not from file

    encoded_image_data = None
    with tf.io.gfile.GFile(fpath, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = b'jpeg' # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmins.append(xmin/iw)
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    xmaxs.append(xmax/iw)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymins.append(ymin/ih)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    ymaxs.append(ymax/ih)

    classes_text = [] # List of string class name of bounding box (1 per box)
    classes_text.append(bytes(clatx,'utf-8'))
    classes = [] # List of integer class id of bounding box (1 per box)
    classes.append(claid)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def writesiglefile(fn,outfile):
    writer = tf.io.TFRecordWriter(outfile)
    file1 = open(fn,'r')
    lines = file1.readlines()
    for ln in lines:
        sts = ln.strip().split(';')
        tf_example = create_tf_example(int(sts[0]),int(sts[1]),float(sts[2]),float(sts[3]),float(sts[4]),float(sts[5]),sts[6],sts[7],int(sts[8]))
        writer.write(tf_example.SerializeToString())
    writer.close()
    file1.close()




def create_tf_example1(imgdata,idx):

    filename = '' # Filename of the image. Empty if image is not from file

    encoded_image_data = None
    with tf.io.gfile.GFile(imgdata.filename, 'rb') as fid:
        encoded_image_data = fid.read()

    key = hashlib.sha256(encoded_image_data).hexdigest()

    image_format = 'jpg' # b'jpeg' or b'png'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tfrecord_lib.convert_to_feature(imgdata.height),
        'image/width': tfrecord_lib.convert_to_feature(imgdata.width),
        'image/filename': tfrecord_lib.convert_to_feature(filename.encode('utf8')),
        'image/source_id': tfrecord_lib.convert_to_feature(str(idx).encode('utf8')),
        'image/key/sha256': tfrecord_lib.convert_to_feature(key.encode('utf8')),
        'image/encoded': tfrecord_lib.convert_to_feature(encoded_image_data),
        'image/format': tfrecord_lib.convert_to_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': tfrecord_lib.convert_to_feature(imgdata.xmin),
        'image/object/bbox/xmax': tfrecord_lib.convert_to_feature(imgdata.xmax),
        'image/object/bbox/ymin': tfrecord_lib.convert_to_feature(imgdata.ymin),
        'image/object/bbox/ymax': tfrecord_lib.convert_to_feature(imgdata.ymax),
        'image/object/class/text': tfrecord_lib.convert_to_feature(imgdata.clsname),
        'image/object/class/label': tfrecord_lib.convert_to_feature(imgdata.clslabel),
        'image/object/is_crowd': tfrecord_lib.convert_to_feature(imgdata.iscrowd),
        'image/object/area': tfrecord_lib.convert_to_feature(imgdata.area, 'float_list'),
    }))
    
    # print(imgdata.area)

    return tf_example


def writesiglefile1(fn,outfile,width):
    imgdatas = {}
    writer = tf.io.TFRecordWriter(outfile)
    file1 = open(fn,'r')
    lines = file1.readlines()
    for ln in lines:
        sts = ln.strip().split(',')
        # ih,iw,xmin,xmax,ymin,ymax,fpath,clatx,claid
        # tf_example = create_tf_example(int(sts[0]),int(sts[1]),float(sts[2]),float(sts[3]),float(sts[4]),float(sts[5]),sts[6],sts[7],int(sts[8]))

        xmin = float(sts[2])
        xmax = float(sts[3])
        ymin = float(sts[4])
        ymax = float(sts[5])
        ih = float(sts[0])
        iw = float(sts[1])

        if sts[6] in imgdatas:
            imgdt = imgdatas[sts[6]]
            imgdt.xmin.append(xmin/iw)
            imgdt.xmax.append(xmax/iw)
            imgdt.ymin.append(ymin/ih)
            imgdt.ymax.append(ymax/ih)
            imgdt.clsname.append(sts[7].encode('utf8'))
            imgdt.clslabel.append(int(sts[8]))
            imgdt.iscrowd.append(0)
            imgdt.area.append((xmax-xmin)*(ymax-ymin))
        else:
            imgdt = imagedata()
            imgdt.height = int(sts[0])
            imgdt.width = int(sts[1])
            imgdt.filename = sts[6]
            imgdt.xmin.append(xmin/iw)
            imgdt.xmax.append(xmax/iw)
            imgdt.ymin.append(ymin/ih)
            imgdt.ymax.append(ymax/ih)
            imgdt.clsname.append(sts[7].encode('utf8'))
            imgdt.clslabel.append(int(sts[8]))
            imgdt.iscrowd.append(0)
            imgdt.area.append((xmax-xmin)*(ymax-ymin))
            imgdatas[sts[6]] = imgdt
    file1.close()
    

    idx = 1
    for k,v in imgdatas.items():
        tf_example = create_tf_example1(v,idx)
        writer.write(tf_example.SerializeToString())
        cimg = cv2.imread(v.filename,cv2.IMREAD_COLOR)
        i = 0
        for xmin in v.xmin:
            cv2.rectangle(cimg,(int(v.xmin[i]*width),int(v.ymin[i]*width)),(int(v.xmax[i]*width),int(v.ymax[i]*width)),(0,255,0),2)
            cx = int(((v.xmin[i]*width+v.xmax[i]*width)/2))
            cy = int(((v.ymin[i]*width+v.ymax[i]*width)/2))
            cv2.putText(cimg,str(int(v.clslabel[i])),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            i = i + 1
        cv2.imwrite('./checkpic/'+str(idx)+'.jpg',cimg)
        idx = idx + 1
    writer.close()


def convertimag4cls(imagepath,srcstr,desstr,size):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        print(fn)
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            img = cv2.imread(fn,cv2.IMREAD_COLOR)
            img = cv2.resize(img,(size,size))
            nfile = fn.upper().replace('.JPEG','.JPG').replace('.PNG','.JPG').replace('.BMP','.JPG').replace(srcstr,desstr)
            cv2.imwrite(nfile,img,[cv2.IMWRITE_JPEG_QUALITY,100])

def tfexample4cls(fn,label,high,width,idx):

    filename = '' # Filename of the image. Empty if image is not from file
    encoded_image_data = None
    with tf.io.gfile.GFile(fn, 'rb') as fid:
        encoded_image_data = fid.read()
    image_format = 'jpg' # b'jpeg' or b'png'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tfrecord_lib.convert_to_feature(high,'int64'),
        'image/width': tfrecord_lib.convert_to_feature(width,'int64'),
        'image/filename': tfrecord_lib.convert_to_feature(filename.encode('utf8')),
        'image/source_id': tfrecord_lib.convert_to_feature(str(idx).encode('utf8')),
        'image/encoded': tfrecord_lib.convert_to_feature(encoded_image_data),
        'image/format': tfrecord_lib.convert_to_feature(image_format.encode('utf8')),
        'image/class/label': tfrecord_lib.convert_to_feature(label,'int64'),
    }))
    return tf_example


def write_coherentvcsel_clstffile(imagepath,high,width,outfile):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    label_names = ['A10','F2X1','F5X1','SIXINCH','ZURICH']
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    writer = tf.io.TFRecordWriter(outfile)
    idx = 0
    for fn in all_image_paths:
        label = all_image_labels[idx]
        print(fn)
        print(label)
        tf_example = tfexample4cls(fn,label,high,width,idx)
        writer.write(tf_example.SerializeToString())
        idx = idx + 1
    writer.close()


def write_boardcomchar_clstffile(imagepath,high,width,outfile):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    label_names = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    writer = tf.io.TFRecordWriter(outfile)
    idx = 0
    for fn in all_image_paths:
        label = all_image_labels[idx]
        print(fn)
        print(label)
        tf_example = tfexample4cls(fn,label,high,width,idx)
        writer.write(tf_example.SerializeToString())
        idx = idx + 1
    writer.close()


def getDatabaseTestData():
    imgstrs = []
    labs = []
    imgtype = ['OGP-circle2168','OGP-sm-iivi','OGP-rect5x1','COGA-6inch','COGA-SR4','OGP-rect2x1','OGP-A10G','OGP-small5x1','OGP-iivi']

    for tp in imgtype:
        with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.chn.ii-vi.net', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
            cursor = conn.cursor()
            sql = "select top 10 [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where imgval = '56' and Revision = '"+tp+"'"
            cursor.execute(sql)
            rows = cursor.fetchall() 
            for row in rows:
                imgstrs.append(str(row[0]))
                labs.append(int(row[1])-48)
            cursor.close()
    idx = 0
    for imgstr in imgstrs:
        bt = base64.b64decode(imgstr)
        im = Image.open(io.BytesIO(bt))
        img1 = np.array(im)
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
        i = img1.flatten().reshape(50,50,3)
        i = i.astype(np.uint8)
        im = Image.fromarray(i)
        im.save('./fonts/eight'+str(idx)+'.png')
        idx = idx + 1


def Convert800GWB(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    idx = 1
    for fn in all_image_paths:
        print(fn)
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            img = cv2.imread(fn,cv2.IMREAD_COLOR);
            img = cv2.resize(img,(680,1024));
            img = cv2.transpose(img);
            img = cv2.flip(img, 1);
            cv2.imwrite('./800GSR8/'+str(idx)+'.JPG',img,[cv2.IMWRITE_JPEG_QUALITY,100])
            idx = idx + 1

# Convert800GWB('./800GSR8_RAW')

def CopyJsonFile():
    for idx in range(16,601):
        nfo = open('./800GSR8/'+str(idx)+'.json','w+')
        orgfo = open('./800GSR8/1.json','r');
        for line in orgfo.readlines():
            if '"imagePath"' in line:
                line = line.replace('1.JPG',str(idx)+'.JPG')
            nfo.write(line)
        nfo.close()
        orgfo.close()

CopyJsonFile()


#getDatabaseTestData()

# write_boardcomchar_clstffile('./BDC_TrainData/Train',64,64,'../VBOARDCOMDATA/train_dataset.tfrecord')

# write_boardcomchar_clstffile('./BDC_TrainData/Valid',64,64,'../VBOARDCOMDATA/valid_dataset.tfrecord')



write_coherentvcsel_clstffile('\\\\cnwx-engsys01\\ShareData\\AI_Data\\VCSEL6',480,480,'./mydata/trainsrcdata/VCLAS_DATA/train_dataset.tfrecord')

write_coherentvcsel_clstffile('\\\\cnwx-engsys01\\ShareData\\AI_Data\\VCSEL6_verify',480,480,'./mydata/trainsrcdata/VCLAS_DATA/valid_dataset.tfrecord')

# convertimag4cls('\\\\wux-engsys01\\PlanningForCast\\VCSEL','VCSEL','VCSEL6',480)

# convertimag3('D:\\desktop\\AOI\\LW PHOTO SAMPLE\\2X400G','D:\\desktop\\AOI\\LW PHOTO SAMPLE\\2X400G_AOI',1024)


# convertimag('D:\\PlanningForCast\\condaenv\\COGA-PAD\\train_org',640,640)
# convertimag('D:\\PlanningForCast\\condaenv\\COGA-PAD\\ver_org',640,640)
# convertimag('D:\\PlanningForCast\\condaenv\\COGA-PAD\\check_org',640,640)
# convertimag('D:\\PlanningForCast\\condaenv\\VISION\\COC_PIC\\IMG',640,640)

# convertimag('D:\\PlanningForCast\\condaenv\\VISION\\zurich\\orginal',640,640)

#XML2CSV()
# writesiglefile1('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\XTRAIN512.txt','../XPADDATA/x_train_dataset.tfrecord')
# writesiglefile1('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\XVERIFY512.txt','../XPADDATA/x_valid_dataset.tfrecord')

# writesiglefile1('./WTRAIN_cla4.txt','../WBDATA/w4_train_dataset.tfrecord')
# writesiglefile1('./WVALID_cla4.txt','../WBDATA/w4_valid_dataset.tfrecord')

# writesiglefile1('./WTRAIN_cla1.txt','../WBDATA/w1_train_dataset.tfrecord')
# writesiglefile1('./WVALID_cla1.txt','../WBDATA/w1_valid_dataset.tfrecord')

# python labelme2coco.py ./mydata ./myoutput --labels labels2.txt 
#python labelme2coco.py ./Train ./TrainOut --labels 800G-BOX-Label.txt
#800G-BOX-Label.txt
# __ignore__
# _background_
# WB1
# WB2
# WB3
# WB4
# WB5



# python create_coco_tf_record.py  --include_masks=True --image_dir=.\myoutput\  --object_annotations_file=./myoutput/annotations.json  --output_file_prefix=./mytraindata --num_shards=1

#python create_coco_tf_record.py  --include_masks=False --image_dir=./TrainOut  --object_annotations_file=./TrainOut/annotations.json   --output_file_prefix=./800GCOCOTrain --num_shards=1
#python create_coco_tf_record.py  --include_masks=False --image_dir=./VerifyOut  --object_annotations_file=./VerifyOut/annotations.json   --output_file_prefix=./800GCOCOVerify --num_shards=1

# python create_coco_tf_record.py  --include_masks=True --image_dir=.\zurich_valid_coco  --object_annotations_file=./zurich_valid_coco/annotations.json   --output_file_prefix=./zurich_valid --num_shards=1
# python create_coco_tf_record.py  --include_masks=True --image_dir=.\zurich_train_coco  --object_annotations_file=./zurich_train_coco/annotations.json   --output_file_prefix=./zurich_train --num_shards=1


