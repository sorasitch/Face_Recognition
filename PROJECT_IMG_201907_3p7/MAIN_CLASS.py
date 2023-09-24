# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:35:08 2020

@author: choms
"""




import os

#import tensorflow as tf
#from tensorflow import keras

#tf.compat.v1.disable_eager_execution()
#tf.config.threading.set_inter_op_parallelism_threads(4)
#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['TF_DISABLE_MKL'] = '1'

#export TF_DISABLE_MKL=1
#os.putenv('TF_DISABLE_MKL', '1')
#print(os.environ['PATH'])
#print(os.environ['TF_DISABLE_MKL'])
# delete the existing values 
#del os.environ["OMP_PROC_BIND'] 
#del os.environ['KMP_BLOCKTIME']
#import tensorflow # this sets KMP_BLOCKTIME and OMP_PROC_BIND 

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
#from statsmodels.api import datasets
from sklearn import datasets ## Get dataset from sklearn
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as nr
import sys
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load


# deep learning
import keras
from keras.datasets import mnist
import keras.utils.np_utils as ku
import keras.models as models
import keras.layers as layers
from keras import regularizers
from keras.layers import Dropout , Reshape
from keras.optimizers import rmsprop, Adam


from keras import optimizers
import numpy.linalg as nll
import time
import matplotlib.pyplot as plt1
import math

#from keras import layers
from keras.datasets import imdb
from keras import preprocessing as prepro
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SimpleRNN, LSTM, GRU, Bidirectional
from keras.layers.normalization import BatchNormalization
from numpy.random import randint
from numpy import argmax

import warnings
from keras.callbacks import Callback

import os.path
import cv2
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
from sklearn.decomposition import PCA
from skimage import io, color
import matplotlib
matplotlib.use('Agg')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage import feature
import os
from keras.utils import to_categorical



from IMGDEEPLERNIG_CLASS import IMGDeepLerning_Class
from READCAPTUREVIDEO_CLASS import ReadCaptureVideo_class
from CAPTUREVIDEO_CLASS import CaptureVideo_class
from DLLIVEPREDICT_CLASS import DLLivePredict_Class

def CaptureVideo():
#    ctv=CaptureVideo_class( owners="SORASIT", characters="FACE")  
#    ctv=CaptureVideo_class( owners="NARES", characters="FACE") 
#    ctv=CaptureVideo_class( owners="PIPAT", characters="FACE") 
#    ctv=CaptureVideo_class( owners="VAL", characters="FACE") 
#    ctv=CaptureVideo_class( owners="UTHIT", characters="FACE")
#    ctv=CaptureVideo_class( owners="Aphiwat", characters="FACE") 
#    ctv=CaptureVideo_class( owners="OUM", characters="FACE") 
    ctv=CaptureVideo_class( owners="UNKNOW", characters="FACE") 
    for x in range(10) :
        print("* Start record : " + str(x) )
        ctv.faceDetectionCapture(
                             numberoffaces=20,
                             h_resize=300,w_resize=300,
                             stop=True
                             )
        print("* Stop record : " + str(x) )
    #ctv.faceDetectionCaptureDebug(
    #                     numberoffaces=20,
    #                     h_resize=300,w_resize=300
    #                     )
    
    
def ReadCaptureVideo_Train():
    rcv=ReadCaptureVideo_class()
    rcv.readPNG()
    #rcv.faceDetectionReadDebug(
    #                         numberoffaces=20,
    #                         h_resize=300,w_resize=300
    #                         )
    
#    img_x_train,img_y_train,img_names,img_json=rcv.faceDetectionRead(
##                                                 numberoffaces=20,
#                                                 h_resize=50,w_resize=50,
#                                                 #"real","gray","local_binary_pattern","Eigenfaces","hsv","edges","laplacian","lbp0","lbp1"
#                                                 imgprocess="real",   
#                                                 showandsave=False
#                                                     )
#    
#    rcv.saveInputWithData(img_names , "real" , img_x_train.shape)
#    img_names=rcv.saveModelNameWithData(img_names , "real" , img_json)
#    print("**TRAIN MODEL**") 
#    print(img_names)
#    img_model = IMGDeepLerning_Class()
#    img_model.X_train=img_x_train #it already be float
#    img_model.Y_train=img_y_train
#    img_model.ModelName_train=img_names
#    img_model.TrainModel() 


#    img_x_train,img_y_train,img_names,img_json=rcv.faceDetectionRead(
##                                                 numberoffaces=20,
#                                                 h_resize=50,w_resize=50,
#                                                 #"real","gray","local_binary_pattern","Eigenfaces","hsv","edges","laplacian","lbp0","lbp1"
#                                                 imgprocess="Eigenfaces",   
#                                                 showandsave=False
#                                                     )
#    
#    rcv.saveInputWithData(img_names , "Eigenfaces" , img_x_train.shape)
#    img_names=rcv.saveModelNameWithData(img_names , "Eigenfaces" , img_json)
#    print("**TRAIN MODEL**") 
#    print(img_names)
#    img_model = IMGDeepLerning_Class()
#    img_model.X_train=img_x_train #it already be float
#    img_model.Y_train=img_y_train
#    img_model.ModelName_train=img_names
#    img_model.TrainModel() 


#    img_x_train,img_y_train,img_names,img_json=rcv.faceDetectionRead(
##                                                 numberoffaces=20,
#                                                 h_resize=50,w_resize=50,
#                                                 #"real","gray","local_binary_pattern","Eigenfaces","hsv","edges","laplacian","lbp0","lbp1"
#                                                 imgprocess="hsv",   
#                                                 showandsave=False
#                                                     )
#    
#    rcv.saveInputWithData(img_names , "hsv" , img_x_train.shape)
#    img_names=rcv.saveModelNameWithData(img_names , "hsv" , img_json)
#    print("**TRAIN MODEL**") 
#    print(img_names)
#    img_model = IMGDeepLerning_Class()
#    img_model.X_train=img_x_train #it already be float
#    img_model.Y_train=img_y_train
#    img_model.ModelName_train=img_names
#    img_model.TrainModel() 
    
    
#    img_x_train,img_y_train,img_names,img_json=rcv.faceDetectionRead(
##                                                 numberoffaces=20,
#                                                 h_resize=50,w_resize=50,
#                                                 #"real","gray","local_binary_pattern","Eigenfaces","hsv","edges","laplacian","lbp0","lbp1"
#                                                 imgprocess="lbp0",   
#                                                 showandsave=False
#                                                     )
#    
#    rcv.saveInputWithData(img_names , "lbp0" , img_x_train.shape)
#    img_names=rcv.saveModelNameWithData(img_names , "lbp0" , img_json)
#    print("**TRAIN MODEL**") 
#    print(img_names)
#    img_model = IMGDeepLerning_Class()
#    img_model.X_train=img_x_train #it already be float
#    img_model.Y_train=img_y_train
#    img_model.ModelName_train=img_names
#    img_model.TrainModel() 
    
    
    img_x_train,img_y_train,img_names,img_json=rcv.faceDetectionRead(
#                                                 numberoffaces=20,
                                                 h_resize=300,w_resize=300,
                                                 #"real","gray","local_binary_pattern","Eigenfaces","hsv","edges","laplacian","lbp0","lbp1"
                                                 imgprocess="gray",   
                                                 showandsave=False
                                                     )
    
    rcv.saveInputWithData(img_names , "gray" , img_x_train.shape)
    img_names=rcv.saveModelNameWithData(img_names , "gray" , img_json)
    print("**TRAIN MODEL**") 
    print(img_names)
    img_model = IMGDeepLerning_Class()
    img_model.X_train=img_x_train #it already be float
    img_model.Y_train=img_y_train
    img_model.ModelName_train=img_names
    img_model.TrainModel() 
    
    
    
    
def ReadCaptureVideo_Test():
    rcv=ReadCaptureVideo_class()
    rcv.readPNG()
    #rcv.faceDetectionReadDebug(
    #                         numberoffaces=20,
    #                         h_resize=300,w_resize=300
    #                         )
    
    img_x_train,img_y_train,img_names,img_json=rcv.faceDetectionRead(
#                                                 numberoffaces=20,
                                                 h_resize=50,w_resize=50,
                                                 #"real","FisherFaces","gray","local_binary_pattern","Eigenfaces","hsv","edges","laplacian","lbp0","lbp1"
                                                 imgprocess="gray",   
                                                 showandsave=False
                                                     )
    
    rcv.saveInputWithData(img_names , "gray" , img_x_train.shape)
    img_names=rcv.saveModelNameWithData(img_names , "gray" , img_json)

    
    img1=IMGDeepLerning_Class()
    print(img_names)
    img1.TestModel(img_names,np.array(img_x_train).astype('float'),np.array(img_y_train).astype('int'))
   
    
#    xtest=img_x_train 
#    ytest=img_y_train
#    print(img_x_train.shape)
#    print(img_y_train.shape)
#    img1.TestModel(img_names,xtest[0:50],ytest[0:50])
    
    
#    xtest=[]  
#    ytest=[] 
#    xtest.extend(img_x_train)
#    ytest.extend(img_y_train)
#    img1.TestModel(img_names,np.array(xtest).astype('float'),np.array(ytest).astype('int'))
    
    
#    pics=cv2.imread(".\\IMGPATH\\SORASIT\\FACE\\161.png")
#    xtest = rcv.faceProcess(
#                                 pics=pics,
#                                 owners="test",
#                                 chars="test",
#                                 idx=0,
#                                 imgprocess="real"
#                                 )
#    b=[]
#    b.append(xtest)
#    bb=np.array(b).astype('float')
#    print(bb.shape)
#    
#    
#    y=[1 , 0]
#    y1=[]
#    y1.append(y)
#    yy=np.array(y1).astype('int')
#    print(bb.shape)
#    print(yy.shape)
#    img1.TestModel(img_names,bb,yy)
    
    
    
    
def ReadCaptureVideo_Load():
    
    print("**LOAD MODEL**") 
    
    ld = []
    ld_labal = []
    ld_owners = []
    ld_chars = []
    ld_models = []
    ld_sizex = []
    ld_sizey = []
    ld_sizez = []
    ld_imgproc = []
    
    
    rcv=ReadCaptureVideo_class()
    rcv.readModelWithData()
    
    i=0
    for labal,owners,chars,model,sizex,sizey,sizez,imgproc in zip(rcv.MODELLABEL,rcv.MODELOWNERS,rcv.MODELCHARS,rcv.MODELS,rcv.MODELSIZEX,rcv.MODELSIZEX,rcv.MODELSIZEZ,rcv.MODELIMGPROCESS) :
        print(labal)
        print(owners)
        print(chars)
        print(model)
        print(sizex)
        print(sizey)
        print(sizez)
        print(imgproc)
        
        ld_labal.append(labal)
        ld_owners.append(owners)
        ld_chars.append(chars)
        ld_models.append(model) 
        ld_sizex.append(sizex) 
        ld_sizey.append(sizey)
        ld_sizez.append(sizez)
        ld_imgproc.append(imgproc)
        
        ld.append(IMGDeepLerning_Class())
        ld[i].Label_load=labal
        ld[i].Owners_load=owners
        ld[i].Chars_load=chars
        ld[i].ModelName_load=model
        ld[i].SizeX_load=sizex
        ld[i].SizeY_load=sizey
        ld[i].SizeZ_load=sizez
        ld[i].IMGProcess_load=imgproc
        ld[i].LoadModel()
        
        i=i+1


#    pics=cv2.imread(".\\IMGPATH\\SORASIT1\\FACE\\1.png")
    pics=cv2.imread(".\\IMGPATH\\SORASIT\\FACE\\161.png")
#    pics=cv2.imread(".\\IMGPATH\\PIPAT\\FACE\\161.png")

   
#    ctv=CaptureVideo_class( owners="TEST", characters="TEST",)  
#    pics=ctv.faceDetection(
#                         numberoffaces=1,
#                         h_resize=300,w_resize=300,
#                         stop=True,
#                         save=True
#                         )
    
    

    final_label = -1
    final_plabel = 0.7
    final_owner = ""
    final_chars = ""
    final_model = ""
    
#    print(xtest.shape)
#    print(len(ld))
#    sys.exit()
    b=[]
    for LD,labal,owners,chars,model,sizex,sizey,sizez,imgproc in zip(ld,ld_labal,ld_owners,ld_chars,ld_models,ld_sizex,ld_sizey,ld_sizez,ld_imgproc) :

        b=[]
        xtest = rcv.faceProcess(
                                 pics=pics,
                                 owners="TEST",
                                 chars="TEST",
                                 idx=0,
#                                 numberoffaces=20,
                                 h_resize=sizey,w_resize=sizex,
                                 imgprocess=str(imgproc),
                                 showandsave=False
                                 )
        
        b.append(xtest)
        bb=np.array(b).astype('float') #needed
        print(bb.shape)
        
        lab,plab=LD.TestLoadModel(bb,0.5)
#        lab,plab=LD.TestLoadModel(b,0.5)
        print(model)
        print(LD.Owners_load[int(lab)])
        print(plab)
        
        
        if plab > final_plabel : 
            final_model= model
            final_label=lab
            final_plabel=plab
            final_owner=LD.Owners_load[int(lab)]
            final_chars=LD.Chars_load[int(lab)]
            
         
    if final_label > -1 : 
        print(final_model)
        print(final_label)
        print(final_plabel)
        print(final_owner)
        print(final_chars)
    else:
        print("Don't know !")

    print("****") 



def LivePredict():
    live=DLLivePredict_Class()
    live.loadDLmodel()
    live.liveFaceDetection(
                            numberoffaces=1,
                            h_resize=300,w_resize=300,
                                 )


#CaptureVideo()    
#ReadCaptureVideo_Train() 
#ReadCaptureVideo_Test()
#ReadCaptureVideo_Load()
LivePredict()
    
    