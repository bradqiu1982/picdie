import pathlib
import os
import cv2
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED
import numpy as np
import sys
import shutil

def convertimag_resize(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    executor = ThreadPoolExecutor(30)
    all_tasks = [executor.submit(writeimg3,fn) for fn in all_image_paths]
    wait(all_tasks,return_when=ALL_COMPLETED)

def writeimg(fn):
    print(fn)
    if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
        img = cv2.imread(fn,cv2.IMREAD_COLOR)
        hg,wd,ch = img.shape
        if(hg > 3000 or wd > 3000):
            nhg = int(float(hg)/6.0)
            nwd = int(float(wd)/6.0)
            img = cv2.resize(img,(nwd,nhg))
        else:
            nhg = int(float(hg)/1.5)
            nwd = int(float(wd)/1.5)
            img = cv2.resize(img,(nwd,nhg))
        cv2.imwrite(fn,img)


def writeimg2(fn):
    print(fn)
    if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
        img = cv2.imread(fn,cv2.IMREAD_COLOR)
        hg,wd,ch = img.shape
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper()):
            nhg = int(float(hg)/2.0)
            nwd = int(float(wd)/2.0)
            img = cv2.resize(img,(nwd,nhg))
        else:
            nhg = int(float(hg)/1.5)
            nwd = int(float(wd)/1.5)
            img = cv2.resize(img,(nwd,nhg))
        cv2.imwrite(fn,img)

def writeimg3(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    idx = 11001
    for fn in all_image_paths:
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            print(fn)
            img = cv2.imread(fn,cv2.IMREAD_COLOR)
            if ('.JPG' in fn.upper() or '.JPEG' in fn.upper()):
                # nhg = 1280
                # nwd = 1280
                # img = cv2.resize(img,(nwd,nhg))
                cv2.imwrite('./trainx/'+str(idx)+'.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,100])
                idx = idx+1

# writeimg3('./train')




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


def ConvertPICX(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    for fn in all_image_paths:
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            print(fn)
            fns = fn.split('\\')
            print(fns[len(fns)-1])

            newimgpath = fn.replace("train","trainx").replace(".jpg","x.jpg")
            cimg = cv2.imread(fn,cv2.IMREAD_COLOR);

            nhg = 1280
            nwd = 1280
            cimg = cv2.resize(cimg,(nwd,nhg))

            cimg = CLAHE(cimg,2.1,16)
            cimg = gamma_correction_lab(cimg,0.9)
            cv2.imwrite(newimgpath,cimg,[cv2.IMWRITE_JPEG_QUALITY,100])


            # orgjspath = fn.replace(".jpg",".json")
            # jspath = fn.replace("train","trainx").replace(".jpg","x.json")
            # jsfn = fns[len(fns)-1].replace(".jpg","x.jpg")

            # nfo = open(jspath,'w+')
            # orgfo = open(orgjspath,'r');
            # for line in orgfo.readlines():
            #     if '"imagePath"' in line:
            #         newline = '  "imagePath": "'+jsfn+'",\r\n'
            #         nfo.write(newline)
            #     else:
            #         nfo.write(line)
            # nfo.close()
            # orgfo.close()


ConvertPICX('./train')


def ConvertPICY(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    for fn in all_image_paths:
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            print(fn)
            fns = fn.split('\\')
            print(fns[len(fns)-1])

            newimgpath = fn.replace("train","trainy").replace(".jpg","y.jpg")
            cimg = cv2.imread(fn,cv2.IMREAD_COLOR);

            cimg = cv2.transpose(cimg);
            cimg = cv2.flip(cimg, 1);
            cimg = cv2.transpose(cimg);
            cimg = cv2.flip(cimg, 1);

            cimg = CLAHE(cimg,2.1,16)
            cimg = gamma_correction_lab(cimg,0.85)

            cv2.imwrite(newimgpath,cimg,[cv2.IMWRITE_JPEG_QUALITY,100])


            orgjspath = fn.replace(".jpg",".json")
            jspath = fn.replace("train","trainy").replace(".jpg","y.json")
            jsfn = fns[len(fns)-1].replace(".jpg","y.jpg")

            nfo = open(jspath,'w+')
            orgfo = open(orgjspath,'r');
            for line in orgfo.readlines():
                if '"imagePath"' in line:
                    newline = '  "imagePath": "'+jsfn+'",\r\n'
                    nfo.write(newline)
                else:
                    nfo.write(line)
            nfo.close()
            orgfo.close()

# ConvertPICY('./train')



def convertimag_resize1(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        print(fn)
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            img = cv2.imread(fn,cv2.IMREAD_COLOR)
            hg,wd,ch = img.shape
            if(hg > 3000 or wd > 3000):
                nhg = int(float(hg)/6.0)
                nwd = int(float(wd)/6.0)
                img = cv2.resize(img,(nwd,nhg))
            else:
                nhg = int(float(hg)/1.5)
                nwd = int(float(wd)/1.5)
                img = cv2.resize(img,(nwd,nhg))
            cv2.imwrite(fn,img)


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


# convertimag_resize('./test')

def Convert800GWB1024(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    idx = 1000
    for fn in all_image_paths:
        print(fn)
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            img = cv2.imread(fn,cv2.IMREAD_COLOR);
            img = cv2.resize(img,(680,1024));
            img = cv2.transpose(img);
            img = cv2.flip(img, 1);
            array_created = np.full((1024, 1024, 3),0, dtype = np.uint8)
            array_created[0:680,0:1024] = img
            grayimg = cv2.cvtColor(array_created,cv2.COLOR_BGR2GRAY)
            cv2.imwrite('./800GSR8_1024_gray/'+str(idx)+'.JPG',grayimg,[cv2.IMWRITE_JPEG_QUALITY,100])
            idx = idx + 1

# Convert800GWB1024('./800GSR8_RAW2')

def Convert800GWB1280(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    idx = 6000
    for fn in all_image_paths:
        print(fn)
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            img = cv2.imread(fn,cv2.IMREAD_COLOR);
            img = cv2.resize(img,(856,1280));
            img = cv2.transpose(img);
            img = cv2.flip(img, 1);
            array_created = np.full((1280, 1280, 3),0, dtype = np.uint8)
            array_created[0:856,0:1280] = img
            grayimg = cv2.cvtColor(array_created,cv2.COLOR_BGR2GRAY)
            cv2.imwrite('./800GSR_gold_gray_1280_4/'+str(idx)+'.JPG',grayimg,[cv2.IMWRITE_JPEG_QUALITY,100])
            idx = idx + 1

# Convert800GWB1280('./800GSR8_RAW7')


# def Conver800GWB1024(imagepath):
#     data_root = pathlib.Path(imagepath)
#     all_image_paths = list(data_root.glob('*'))
#     all_image_paths = [str(path) for path in all_image_paths]
#     for fn in all_image_paths:
#         if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
#             img = cv2.imread(fn,cv2.IMREAD_COLOR);
#             array_created = np.full((1024, 1024, 3),0, dtype = np.uint8)
#             array_created[0:680,0:1024] = img
#             cv2.imwrite(fn,array_created,[cv2.IMWRITE_JPEG_QUALITY,100])

#Conver800GWB1024('./800GSR8_wb/Test')

def Conver800GWBJSON(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        if '.JSON' in fn.upper():
            nfo = open(fn.replace('_1024','_jason'),'w+')
            orgfo = open(fn,'r');
            for line in orgfo.readlines():
                if '"imageHeight": 680' in line:
                    line = line.replace('680','1024')
                nfo.write(line)
            nfo.close()
            orgfo.close()
#Conver800GWBJSON('./800GSR8_wb/Test_1024')

def CopyJsonFile():
    try:
        for idx in range(7600,7846):
            nfo = open('./240621-103/'+str(idx)+'.json','w+')
            orgfo = open('./240621-103/1.json','r');
            for line in orgfo.readlines():
                if '"imagePath"' in line:
                    line = line.replace('1.JPG',str(idx)+'.JPG')
                nfo.write(line)
            nfo.close()
            orgfo.close()
    except:
        exception_message = sys.exc_info()[1]
        print(str(exception_message))

# CopyJsonFile()


def EnhanceImg(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            print(fn)
            img = cv2.imread(fn,cv2.IMREAD_COLOR);
            img = cv2.detailEnhance(img)
            cv2.imwrite(fn,img,[cv2.IMWRITE_JPEG_QUALITY,100])

# EnhanceImg('./800GSR_gold_1280_enhance')
# EnhanceImg('./800GSR_gold_gray_1280_enhance')

def EnhanceImg2(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            print(fn)
            img = cv2.imread(fn,cv2.IMREAD_COLOR);
            img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
            cv2.imwrite(fn,img,[cv2.IMWRITE_JPEG_QUALITY,100])

# EnhanceImg2('./enh1')


def AddL4L5(imagepath):
    appendstr = """
    {
      "label": "L5",
      "points": [
        [
          418.2051282051282,
          358.8034188034188
        ],
        [
          403.6752136752136,
          340.85470085470087
        ],
        [
          296.8376068376068,
          382.7350427350427
        ],
        [
          298.54700854700855,
          422.05128205128216
        ],
        [
          319.9145299145299,
          441.7094017094018
        ],
        [
          413.0769230769231,
          382.7350427350427
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "polygon",
      "flags": {},
      "mask": null
    },
    {
      "label": "L4",
      "points": [
        [
          951.5384615384615,
          398.11965811965814
        ],
        [
          954.957264957265,
          376.7521367521368
        ],
        [
          1037.0085470085469,
          292.13675213675214
        ],
        [
          1057.521367521368,
          309.2307692307692
        ],
        [
          1053.2478632478637,
          351.96581196581195
        ],
        [
          960.0854700854701,
          419.4871794871795
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "polygon",
      "flags": {},
      "mask": null
    },
    """

    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        if ('.JSON' in fn.upper()):
            alllines = []
            orgfo = open(fn,'r');
            for line in orgfo.readlines():
                alllines.append(line)
            orgfo.close()

            nfo = open(fn,'w+')
            for line in alllines:
                if '"shapes": [' in line:
                    nfo.write(line)
                    nfo.write(appendstr)
                else:
                    nfo.write(line)
            nfo.close()


# AddL4L5('./yinsheng/train')
# AddL4L5('./yinsheng/verify')


def ConvertSIPH_COC(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*/*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    idx = 2000
    for fn in all_image_paths:
        print(fn)
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            img = cv2.imread(fn,cv2.IMREAD_COLOR);
            img = cv2.resize(img,(1280,848));
            array_created = np.full((1280, 1280, 3),0, dtype = np.uint8)
            array_created[0:848,0:1280] = img
            cv2.imwrite('./siph-coc-new/'+str(idx)+'.JPG',array_created,[cv2.IMWRITE_JPEG_QUALITY,100])
            idx = idx + 1

# ConvertSIPH_COC('./siph-coc72')



def CopyJsonFile2():
    try:
        for idx in range(2001,2035):
            nfo = open('./siph-coc-new/'+str(idx)+'.json','w+')
            orgfo = open('./siph-coc-new/2000.json','r');
            for line in orgfo.readlines():
                if '"imagePath"' in line:
                    line = line.replace('2000.JPG',str(idx)+'.JPG')
                nfo.write(line)
            nfo.close()
            orgfo.close()
    except:
        exception_message = sys.exc_info()[1]
        print(str(exception_message))


# CopyJsonFile2()


def ConvertEMLSWD_COC(imagepath):
    data_root = pathlib.Path(imagepath)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    for fn in all_image_paths:
        if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
            print(fn)
            # img = cv2.imread(str(fn.encode('utf8').decode('utf8')),cv2.IMREAD_COLOR);
            img = cv2.imdecode(np.fromfile(fn),cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img,(1280,960));
            array_created = np.full((1280, 1280, 3),0, dtype = np.uint8)
            array_created[0:960,0:1280] = img

            newfn = fn.replace('800GDR8','800GDR8NEW').replace('芯片压伤','_XPYS').replace('载体崩缺','_ZTBQ').replace('线损伤','_XSS').replace('金线断$线塌','_JXD').replace('芯片破损','_XPPS').replace('芯片脏污','_XPZW').replace('芯片划伤','_XPHS').replace('氮化物析出','DHW').replace('芯片波导破损','_BDPS').replace('波导破损','_BDPS').replace('波导脏污','_BDZW').replace('焊点不良','_HDBL').replace('芯片裂纹','_XPLW').replace('来料芯片不良','_LLBL')

            cv2.imwrite(newfn,array_created,[cv2.IMWRITE_JPEG_QUALITY,100])
            # idx = idx + 1

# ConvertEMLSWD_COC('./800GDR8')



def CopyJsonFile3():
    try:
        pts = []
        pts.append('./V6-0920new')

        idx = 4100
        for pt in pts:
            data_root = pathlib.Path(pt)
            all_image_paths = list(data_root.glob('*'))
            all_image_paths = [str(path) for path in all_image_paths]
            for fn in all_image_paths:
                if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
                    print(fn)
                    nfo = open(pt+'/'+str(idx)+'.json','w+')
                    orgfo = open(pt+'/1.json','r');
                    for line in orgfo.readlines():
                        if '"imagePath"' in line:
                            line = line.replace('1.JPG',str(idx)+'.JPG')
                        nfo.write(line)
                    nfo.close()
                    orgfo.close()
                    idx = idx + 1
    except:
        exception_message = sys.exc_info()[1]
        print(str(exception_message))


# CopyJsonFile3()


def CopyJsonFile4():
    try:
        for idx in range(13501,13806):
            nfo = open('./trainx/'+str(idx)+'x.json','w+')
            orgfo = open('./trainx/13500x.json','r');
            for line in orgfo.readlines():
                if '"imagePath"' in line:
                    line = line.replace('13500x.jpg',str(idx)+'x.jpg')
                nfo.write(line)
            nfo.close()
            orgfo.close()
    except:
        exception_message = sys.exc_info()[1]
        print(str(exception_message))

# CopyJsonFile4()

# def CopySWDFile():
#     pts = []
#     pts.append('./V6-0920')
#     idx = 4100
#     for pt in pts:
#         data_root = pathlib.Path(pt)
#         all_image_paths = list(data_root.glob('*'))
#         all_image_paths = [str(path) for path in all_image_paths]
#         for fn in all_image_paths:
#             if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
#                 print(fn)
#                 img = cv2.imread(fn,cv2.IMREAD_COLOR);
#                 cv2.imwrite(pt.replace('V6-0920','V6-0920new')+'/'+str(idx)+'.JPG',img ,[cv2.IMWRITE_JPEG_QUALITY,100])
#                 idx = idx + 1

# CopySWDFile()

