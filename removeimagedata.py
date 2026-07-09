
import pathlib
import os
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED
import numpy as np
import sys
import shutil


def NOIMGData(imagepath):
    try:
        data_root = pathlib.Path(imagepath)
        all_image_paths = list(data_root.glob('*'))
        all_image_paths = [str(path) for path in all_image_paths]

        for fn in all_image_paths:
            if '.JSON' in fn.upper() :
                print(fn)
                
                wholeline = '';
                orgfo = open(fn,'r');
                for line in orgfo.readlines():
                    if '"imageData"' in line:
                        wholeline += '  "imageData": null,\n'
                    else:
                    	wholeline += line
                orgfo.close()

                nfo = open(fn,'w+')
                nfo.write(wholeline)
                nfo.close()
    except:
        exception_message = sys.exc_info()[1]
        print(str(exception_message))

NOIMGData('./tmp')
