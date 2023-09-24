# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:36:30 2020

@author: schomsin
"""
import os
#export TF_DISABLE_MKL=1
#os.putenv('TF_DISABLE_MKL', '1')
# delete the existing values 
#del os.environ['OMP_PROC_BIND'] 
#del os.environ['KMP_BLOCKTIME']
#import tensorflow # this sets KMP_BLOCKTIME and OMP_PROC_BIND 

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
from sklearn.decomposition import PCA
from skimage import io, color
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage import feature

import sys
import os
import numpy as np

from keras.utils import to_categorical
import pandas as pd

    
    
    
class CaptureVideo_class():
    
    def __init__(self,
                 owners="",
                 characters="",
                 ):
        self.folder_path=".\\IMGPATH"
        self.owners=owners
        self.characters=characters
        self.names=""
        self.__files=""


    def savefiles(self ,
                  faceno=0
                  ):
        pass
        path=self.folder_path
        if(not os.path.exists(path)):
            os.mkdir(path)
    
        owners_path=path+"\\"+self.owners
        if(not os.path.exists(owners_path)):
            os.mkdir(owners_path)
         
        characters_path=owners_path+"\\"+self.characters
        if(not os.path.exists(characters_path)):
            os.mkdir(characters_path)
        
        dirs = os.listdir(characters_path)
        
        n=0
        for names in dirs:
            nn=int(names.replace(".png", ""))
            if n < int(nn) :
                n=int(nn)
            
        n=n+1
        self.__files=characters_path+"\\"+str(n)+".png"
        self.names=characters_path+"\\"+str(n)
        
        
        return self.names
    
    
    def saveDebug(self,
                  faceno=0
                  ):
        pass
        path=".\\IMGDEBUG"
        if(not os.path.exists(path)):
            os.mkdir(path)
    
        owners_path=path+"\\"+self.owners
        if(not os.path.exists(owners_path)):
            os.mkdir(owners_path)
         
        characters_path=owners_path+"\\"+self.characters
        if(not os.path.exists(characters_path)):
            os.mkdir(characters_path)
        
        dirs = os.listdir(characters_path)
        
        n=0
        for names in dirs:
            if not ("_debug.png" in names):
                continue;
            nn=int(names.replace("_debug.png", ""))
            if n < int(nn) :
                n=int(nn)
        n=n+1

        names=characters_path+"\\"+str(n)
        
        return names

    
    def faceDetectionCapture(self,
                             numberoffaces=20,
                             h_resize=300,w_resize=300,
                             stop=False
                             ):
        
        flag = False
        
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
                
                filename=self.savefiles(faceno=f)
                cv2.imwrite(filename+".png",res)
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                
                yy=int(Y[f,0])
                
                if yy != (nF-1) :
                    cv2.putText(frame, "face : "+str(f)+" , count : "+str(yy) , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)

                if yy == (nF-1) :
                    cv2.putText(frame, "face : "+str(f)+" , count : "+str(yy)+" completed"  , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)    
                    Y[f,0]=-1
                    flag=True
              
                Y[f,0]=Y[f,0]+1           
                f=f+1
                eyes = eye_cascade.detectMultiScale(roi_gray)
        #        print(eyes)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.imshow('frame1',frame)  
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if stop and flag :
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
        
    def faceDetectionCaptureDebug(self,
                             numberoffaces=20,
                             h_resize=300,w_resize=300
                                  ):
                                                    
        
        path=".\\FACE"
        if(not os.path.exists(path)):
            os.mkdir(path)
        
        #cap = cv2.VideoCapture('rtsp://192.168.1.36:8554/mainstream')
        #cap = cv2.VideoCapture('rtsp://192.168.1.36:8554/substream')
        cap = cv2.VideoCapture(0)
        
        face_cascade = cv2.CascadeClassifier(".\\haarcascades\\haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(".\\haarcascades\\haarcascade_eye.xml")
        
        
        ## settings for LBP
        radius = 10
        n_points = 30
#        METHOD = 'uniform'
        
        height, width = h_resize,w_resize
        nF  = numberoffaces  # number of pictures

        A = np.zeros([50,nF, height * width]) #50 faces

        Y = np.zeros([50,1]) #50 faces
        pca = PCA(n_components=4)
        
        B = np.zeros([50, 4 ,height , width]) #50 faces
        
        while(True):
            ret, frame = cap.read()
        #    cv2.imshow('frame',frame)  
                      
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#            cv2.imshow('gray',gray)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #    print(faces)
        #    print(len(faces))
        #    if(len(faces)==1):
        #        fx,fy,fw,fh = faces[0]
        #        print(fx,fy,fw,fh)
            f=0
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
        #        height, width = 300,300 #roi_color.shape[:2]
                res = cv2.resize(roi_color,(width, height), interpolation = cv2.INTER_CUBIC)
                
                
                
        #        similar to FisherFaces
                yy=int(Y[f,0])
                A[f ,yy, :]=color.rgb2gray(res).reshape([1, height * width])

                if yy == (nF-1) :

                    pca.fit(A[f , : , : ])
        #            fig = plt.figure(figsize=(14, 14))
                    fig = plt.figure()
                    for i in range(4):
                        ax = plt.subplot(2,2,i + 1)
                        ax.axis('off')
                        B[f, i ,:,:]=pca.components_[i].reshape([width, height])
                        ax.imshow(pca.components_[i].reshape([width, height]))

                    fig.savefig(".\\FACE\\f"+str(f)+"_"+str(i)+".png")
        #           plt.close(fig)
                    img0 = cv2.imread(".\\FACE\\f"+str(f)+"_"+str(i)+".png")
                    cv2.imshow("Eigenfaces"+str(i),img0)
    
                    resgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                    lbp = local_binary_pattern(resgray, n_points, radius,  method='default')
                    cv2.imshow('lbp',lbp)
                    cv2.imwrite(".\\FACE\\f"+str(f)+"_"+"lbp"+".png",lbp)
            
        #            similar to Eigenfaces 
                    ax1 = plt.subplot()
                    ax1.axis('off')
                    imgplot1 = ax1.imshow(resgray)
                    imgplot1.set_cmap('nipy_spectral')
                    fig.savefig(".\\FACE\\f"+str(f)+"_"+"nipy_spectral"+".png")
                    plt.close(fig)
                    img1 = cv2.imread(".\\FACE\\f"+str(f)+"_"+"nipy_spectral"+".png")
                    cv2.imshow("nipy_spectral",img1)
            
            #        resgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            #        hist = desc.describe(resgray)
            ##        print(hist.shape)
                    
            #        similar to Eigenfaces       
                    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
            #        hsvresgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('hsv',hsv)
                    cv2.imwrite(".\\FACE\\f"+str(f)+"_"+"hsv"+".png",hsv)
                
                
                    edges = cv2.Canny(res,50,100)
                    cv2.imshow('edges',edges)
                    cv2.imwrite(".\\FACE\\f"+str(f)+"_"+"edges"+".png",edges)
                
                
                    laplacian = cv2.Laplacian(res,cv2.CV_64F)
                    cv2.imshow('laplacian',laplacian)
                    cv2.imwrite(".\\FACE\\f"+str(f)+"_"+"laplacian"+".png",laplacian)
                
                
            #        similar to LocalBinaryPatterns
            #        resgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                    sobelx = cv2.Sobel(resgray,cv2.CV_64F,1,0,ksize=5)
                    cv2.imshow('sobelx',sobelx)
                    cv2.imwrite(".\\FACE\\f"+str(f)+"_"+"sobelx"+".png",sobelx)
                    
                    ret,thresh1 = cv2.threshold(sobelx,127,255,cv2.THRESH_TOZERO)
                    cv2.imshow('thresh1',thresh1)
                    cv2.imwrite(".\\FACE\\f"+str(f)+"_"+"thresh1"+".png",thresh1)
                    
                    
                    cv2.imshow('roi_color'+ str(f),res)
                    cv2.imwrite(".\\FACE\\f"+str(f)+"_"+"res"+".png",res)         
                            
                        
                    Y[f,0]=-1
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
        
        
    def faceDetection(self,
                             numberoffaces=20,
                             h_resize=300,w_resize=300,
                             stop=False,
                             save=False
                             ):
        
        flag = False
        
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
                
                if save :
                    filename=self.saveDebug(faceno=f)
                    cv2.imwrite(filename+"_debug.png",res)
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                
                yy=int(Y[f,0])
                
                if yy != (nF-1) :
                    cv2.putText(frame, "face : "+str(f)+" , count : "+str(yy) , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)

                if yy == (nF-1) :
                    cv2.putText(frame, "face : "+str(f)+" , count : "+str(yy)+" completed"  , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)    
                    Y[f,0]=-1
                    flag=True
              
                Y[f,0]=Y[f,0]+1           
                f=f+1
                eyes = eye_cascade.detectMultiScale(roi_gray)
        #        print(eyes)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.imshow('frame1',frame)  
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if stop and flag :
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return res
        
        
        
#ctv=CaptureVideo_class( owners="test", characters="test",)  
#for x in range(10) :
#    print("* Start record : " + str(x) )
#    ctv.faceDetectionCapture(
#                         numberoffaces=20,
#                         h_resize=300,w_resize=300,
#                         stop=True
#                         )
#    print("* Stop record : " + str(x) )
#ctv.faceDetectionCaptureDebug(
#                     numberoffaces=1,
#                     h_resize=300,w_resize=300
#                     )

