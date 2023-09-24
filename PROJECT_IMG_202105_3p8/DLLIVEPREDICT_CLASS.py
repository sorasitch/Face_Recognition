# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:53:33 2020

@author: schomsin
"""
import os
#export TF_DISABLE_MKL=1
#os.putenv('TF_DISABLE_MKL', '1')
# delete the existing values 
#del os.environ['OMP_PROC_BIND'] 
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
from keras.optimizers import Adam


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

from collections import Counter

from IMGDEEPLERNIG_CLASS import IMGDeepLerning_Class
from READCAPTUREVIDEO_CLASS import ReadCaptureVideo_class
from CAPTUREVIDEO_CLASS import CaptureVideo_class


class DLLivePredict_Class():
    
    
    def __init__(self
                 ):
            self.ld = []
            self.ld_labal = []
            self.ld_owners = []
            self.ld_chars = []
            self.ld_models = []
            self.ld_sizex = []
            self.ld_sizey = []
            self.ld_sizez = []
            self.ld_imgproc = []
    
    def loadDLmodel(self
                 ):
        
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
                
                self.ld_labal.append(labal)
                self.ld_owners.append(owners)
                self.ld_chars.append(chars)
                self.ld_models.append(model) 
                self.ld_sizex.append(sizex) 
                self.ld_sizey.append(sizey)
                self.ld_sizez.append(sizez)
                self.ld_imgproc.append(imgproc)
                
                self.ld.append(IMGDeepLerning_Class())
                self.ld[i].Label_load=labal
                self.ld[i].Owners_load=owners
                self.ld[i].Chars_load=chars
                self.ld[i].ModelName_load=model
                self.ld[i].SizeX_load=sizex
                self.ld[i].SizeY_load=sizey
                self.ld[i].SizeZ_load=sizez
                self.ld[i].IMGProcess_load=imgproc
                self.ld[i].LoadModel()
                
                i=i+1
                
                
    def testDLmodel(self,
                    pics=[]
                 ):
        
        
        rcv=ReadCaptureVideo_class()
        
        final_label = -1
        final_plabel = 0.9
        final_owner = ""
        final_chars = ""
        final_model = ""
        
    #    print(xtest.shape)
    #    print(len(ld))
    #    sys.exit()
        b=[]
        for LD,labal,owners,chars,model,sizex,sizey,sizez,imgproc in zip(self.ld,self.ld_labal,self.ld_owners,self.ld_chars,self.ld_models,self.ld_sizex,self.ld_sizey,self.ld_sizez,self.ld_imgproc) :
    
            if("lbp0" in model):
                continue;
            if("gray" in model):
                continue;
            if("hsv" in model):
                continue;
            if("Eigenfaces" in model):
                continue;
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
        
        return final_owner,final_chars
     

    def testDLmodel1(self,
                    pics=[]
                 ):
        
        
        rcv=ReadCaptureVideo_class()
        
        final_label = -1
        final_plabel = 0.7
        final_owner = ""
        final_chars = ""
        final_model = ""
        
        final_label_lt = []
        final_plabel_lt = []
        final_owner_lt = []
        final_chars_lt = []
        final_model_lt = []
        
        
    #    print(xtest.shape)
    #    print(len(ld))
    #    sys.exit()
        b=[]
        for LD,labal,owners,chars,model,sizex,sizey,sizez,imgproc in zip(self.ld,self.ld_labal,self.ld_owners,self.ld_chars,self.ld_models,self.ld_sizex,self.ld_sizey,self.ld_sizez,self.ld_imgproc) :
    
#            if("lbp0" in model):
#                continue;
#            if("gray" in model):
#                continue;
#            if("hsv" in model):
#                continue;
#            if("Eigenfaces" in model):
#                continue;
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
            print(LD.Owners_load[int(lab)])
            print(plab)
            
                
            if plab > final_plabel : 
                final_model_lt.append(model)
                final_label_lt.append(int(lab))
                final_plabel_lt.append(plab)
                final_owner_lt.append(LD.Owners_load[int(lab)])
                final_chars_lt.append(LD.Chars_load[int(lab)])
            else :
                final_model_lt.append("")
                final_label_lt.append(-1)
                final_plabel_lt.append(-1)
                final_owner_lt.append("")
                final_chars_lt.append("")
                
                
#        print(final_label_lt)
        c=Counter(final_label_lt)
#        print(c.values())
#        print(c.keys())
        maxValue=-1
        maxKey=-1
        for key, value in c.items():
        #    print(key)
        #    print(value)
            if value>=maxValue :
                maxValue=value
                maxKey=key
                
#        print(maxValue)
#        print(maxKey)  
        if (( maxValue/len(final_label_lt) ) > 0.5) and ( maxKey > -1 ) : 
            final_label=maxKey
            for label , plabel , model in zip(final_model_lt,final_plabel_lt,final_model_lt) :
                if (label==maxKey):
                    final_plabel=plabel
                    final_model=model
                    break
            final_owner=LD.Owners_load[int(maxKey)]
            final_chars=LD.Chars_load[int(maxKey)]
            print(final_label)
            print(final_plabel)
            print(final_owner)
            print(final_chars)
        else :
            print("Don't know !")
        print("****")                 
        
        return final_owner,final_chars
 
        
    def callDLThread(self,
                     pics=[]
                     ):
#        final_owner,final_chars=self.testDLmodel(pics=pics)
        final_owner,final_chars=self.testDLmodel1(pics=pics)
        return final_owner,final_chars
        
        
    def liveFaceDetection(self,                             
                             numberoffaces=20,
                             h_resize=300,w_resize=300,
                             ):

        
        #cap = cv2.VideoCapture('rtsp://192.168.1.36:8554/mainstream')
        #cap = cv2.VideoCapture('rtsp://192.168.1.36:8554/substream')
        cap = cv2.VideoCapture(0)

        
        #face_cascade = cv2.CascadeClassifier('C:\Users\choms\Anaconda3\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
        #eye_cascade = cv2.CascadeClassifier('C:\Users\choms\Anaconda3\Library\etc\haarcascades\haarcascade_eye.xml')
        
        face_cascade = cv2.CascadeClassifier(".\\haarcascades\\haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(".\\haarcascades\\haarcascade_eye.xml")
        
        height, width = h_resize,w_resize
        nF  = numberoffaces  # number of pictures
        #Y=0
        Y = np.zeros([50,1]) #50 faces
        
        while(True):
            ret, frame = cap.read()
        #    cv2.imshow('frame',frame)  
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #    cv2.imshow('gray',gray)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #    print(faces)
        #    print(len(faces))
        #    if(len(faces)==1):
        #        fx,fy,fw,fh = faces[0]
        #        print(fx,fy,fw,fh)
        
            f=0 # number of detected face
            for (x,y,w,h) in faces:
                
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
        #        height, width = 300,300 #roi_color.shape[:2]
                res = cv2.resize(roi_color,(width, height), interpolation = cv2.INTER_CUBIC)
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                
                yy=int(Y[f,0])
                
                if yy != (nF-1) :
                    pass
#                    cv2.putText(frame, "face : "+str(f)+" , count : "+str(yy) , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)

                if yy == (nF-1) :
                    pass
#                    cv2.putText(frame, "face : "+str(f)+" , count : "+str(yy)+" completed"  , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)    
                    Y[f,0]=-1
                    print("call thread here")
                    final_owner,final_chars=self.testDLmodel(res)
#                    final_owner,final_chars=self.callDLThread(res)
                    cv2.putText(frame, str(final_owner)+" "+str(final_chars) , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 1) 
                    
              
                Y[f,0]=Y[f,0]+1           
                f=f+1
                eyes = eye_cascade.detectMultiScale(roi_gray)
        #        print(eyes)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.imshow('frame1',frame)  
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    def test_steaming(self,
                      frame,
                      numberoffaces=20,
                      h_resize=300,w_resize=300):
        height, width = h_resize,w_resize
        nF  = numberoffaces  # number of pictures
        return frame
        
        
    def liveFaceDetection_websteaming(self,   
                                      frame,
                                      numberoffaces=1,
                                      h_resize=300,
                                      w_resize=300
                                      ):

        
        #cap = cv2.VideoCapture('rtsp://192.168.1.36:8554/mainstream')
        #cap = cv2.VideoCapture('rtsp://192.168.1.36:8554/substream')
#        cap = cv2.VideoCapture(0)

        
#        face_cascade = cv2.CascadeClassifier('C:\Users\choms\Anaconda3\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
#        eye_cascade = cv2.CascadeClassifier('C:\Users\choms\Anaconda3\Library\etc\haarcascades\haarcascade_eye.xml')
        
        face_cascade = cv2.CascadeClassifier(".\\haarcascades\\haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(".\\haarcascades\\haarcascade_eye.xml")
        
        height, width = h_resize,w_resize
        nF  = numberoffaces  # number of pictures
        #Y=0
        Y = np.zeros([50,1]) #50 faces
#        cap=cap
#        face_cascade=face_cascade
#        eye_cascade=eye_cascade
#        Y=Y
        #while(True):
#        ret, frame = cap.read()

    #    cv2.imshow('frame',frame)  
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #    cv2.imshow('gray',gray)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #    print(faces)
    #    print(len(faces))
    #    if(len(faces)==1):
    #        fx,fy,fw,fh = faces[0]
    #        print(fx,fy,fw,fh)
    
        f=0 # number of detected face
        for (x,y,w,h) in faces:
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
    #        height, width = 300,300 #roi_color.shape[:2]
            res = cv2.resize(roi_color,(width, height), interpolation = cv2.INTER_CUBIC)
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
            yy=int(Y[f,0])
            
            if yy != (nF-1) :
                pass
#                    cv2.putText(frame, "face : "+str(f)+" , count : "+str(yy) , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)

            if yy == (nF-1) :
                pass
#                    cv2.putText(frame, "face : "+str(f)+" , count : "+str(yy)+" completed"  , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)    
                Y[f,0]=-1
                print("call thread here")
                final_owner,final_chars=self.testDLmodel(res)
#                    final_owner,final_chars=self.callDLThread(res)
                cv2.putText(frame, str(final_owner)+" "+str(final_chars) , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 1) 
                
          
            Y[f,0]=Y[f,0]+1           
            f=f+1
            eyes = eye_cascade.detectMultiScale(roi_gray)
    #        print(eyes)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#        cv2.imshow('frame1',frame)  
        return frame
    
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
        
#        cap.release()
#        cv2.destroyAllWindows()
        
#live=DLLivePredict_Class()
#live.loadDLmodel()
#live.liveFaceDetection(
#                        numberoffaces=1,
#                        h_resize=300,w_resize=300,
#                             )