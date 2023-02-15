import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform
from skimage import img_as_ubyte
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import time
import datetime
from utils import *

if __name__ == "__main__":
    path = './data'
    list_of_tif = []
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            # if filename.endswith('.tif'): 
            list_of_files[filename] = os.sep.join([dirpath, filename])
            list_of_tif.append(filename)

    # SSD & NCC
    date_time = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
    folder_name = f"./results/{date_time}_SSD_NCC"
    try:
        os.mkdir(folder_name)
    except:
        pass
    RGB_synthesis(folder_name, 'cathedral.jpg',square_limit = 25, CROP = False,canny = True, pyramid= False, CMEAS = False, method = 'SSD', min_xx = -100, min_yy = -100, max_xx = 100, max_yy = 100)
    RGB_synthesis(folder_name, 'cathedral.jpg',square_limit = 25, CROP = False,canny = True, pyramid= False, CMEAS = False, method = 'NCC', min_xx = -100, min_yy = -100, max_xx = 100, max_yy = 100)
    # Pyramid with NCC
    date_time = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
    folder_name = f"./results/{date_time}_Pyramid_with_NCC"
    try:
        os.mkdir(folder_name)
    except:
        pass
    for i in range(len(list_of_tif)):
        if i==0 :
            RGB_synthesis(folder_name, list_of_tif[i],square_limit = 25, CROP = False,canny = False, pyramid= True, CMEAS = False, method = 'NCC', min_xx = -100, min_yy = -100, max_xx = 100, max_yy = 100)
        else:
            RGB_synthesis(folder_name, list_of_tif[i],square_limit = 50, CROP = False,canny = False, pyramid= True, CMEAS = False, method = 'NCC', min_xx = -100, min_yy = -100, max_xx = 100, max_yy = 100)
    # AutoCrop
    date_time = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
    folder_name = f"./results/{date_time}_AutoCrop"
    try:
        os.mkdir(folder_name)
    except:
        pass
    for i in range(len(list_of_tif)):
        if i==0 :
            RGB_synthesis(folder_name, list_of_tif[i],square_limit = 25, CROP = True,canny = False, pyramid= True, CMEAS = False, method = 'NCC', min_xx = -100, min_yy = -100, max_xx = 100, max_yy = 100)
        else:
            RGB_synthesis(folder_name, list_of_tif[i],square_limit = 50, CROP = True,canny = False, pyramid= True, CMEAS = False, method = 'NCC', min_xx = -100, min_yy = -100, max_xx = 100, max_yy = 100)
    # Canny Edge Detector
    date_time = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
    folder_name = f"./results/{date_time}_CannyEdgeDetector"
    try:
        os.mkdir(folder_name)
    except:
        pass
    for i in range(len(list_of_tif)):
        if i==0 :
            RGB_synthesis(folder_name, list_of_tif[i],square_limit = 25, CROP = True,canny = True, pyramid= True, CMEAS = False, method = 'NCC', min_xx = -100, min_yy = -100, max_xx = 100, max_yy = 100)
        else:
            RGB_synthesis(folder_name, list_of_tif[i],square_limit = 50, CROP = True,canny = True, pyramid= True, CMEAS = False, method = 'NCC', min_xx = -100, min_yy = -100, max_xx = 100, max_yy = 100)
    # CMEAS Evolution Method
    date_time = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
    folder_name = f"./results/{date_time}_CMEAS"
    try:
        os.mkdir(folder_name)
    except:
        pass
    for i in range(len(list_of_tif)):
        if i==0 :
            RGB_synthesis(folder_name, list_of_tif[i],square_limit = 25, CROP = True,canny = True, pyramid= True, CMEAS = True, method = 'NCC', min_xx = -100, min_yy = -100, max_xx = 100, max_yy = 100)
        else:
            RGB_synthesis(folder_name, list_of_tif[i],square_limit = 50, CROP = True,canny = True, pyramid= True, CMEAS = True, method = 'NCC', min_xx = -100, min_yy = -100, max_xx = 100, max_yy = 100)

