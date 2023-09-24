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

import io
import json

    
    
    
class ReadCaptureVideo_class():
    
    def __init__(self
                 ):
        self.folder_path=".\\IMGPATH"
        
        self.OWNERS=[]
        self.CHARACTERS=[]
        self.files_list=[]
        
        self.__X_DATASET=[]
        self.__X_DATASET0=[]
        self.X_DATASET=[]
        
        self.__MODELOWNERS = []
        self.__MODELCHARS = []
        self.__MODELLABEL = []
        self.MODELS = []
        self.MODELOWNERS = []
        self.MODELCHARS = []
        self.MODELLABEL = []
        self.MODELSIZEX = []
        self.MODELSIZEY = []
        self.MODELSIZEZ = []
        self.MODELIMGPROCESS = []

    def readPNG(self):
        
        
        files=""
        path=self.folder_path
        dirs = os.listdir(path)
#        print(dirs)
        
        for owners in dirs:
            pass
            self.OWNERS.append(owners)
            owners_path = path + "\\" + owners
            dirs1 = os.listdir(owners_path)
            print(owners_path)
            
            self.__X_DATASET=[]
            for chars in dirs1:
                pass
                self.CHARACTERS.append(chars)
                characters_path = owners_path + "\\" + chars
                dirs2 = os.listdir(characters_path)
                print(characters_path)

                self.__X_DATASET0=[]
                for pics in dirs2:
                    pass
                    files = characters_path + "\\" + pics
                    self.__X_DATASET0.append(cv2.imread(files))
                    self.files_list.append(files)
#                    print(files) 
                self.__X_DATASET.append(self.__X_DATASET0)
                
            self.X_DATASET.append(self.__X_DATASET) 
        
        if(len(self.files_list)>0):
            pass
            return self.files_list
            
        return files
       
        
    def faceDetectionRead(self,
                             numberoffaces=20,
                             h_resize=300,w_resize=300,
                             imgprocess="real",
                             showandsave=False
                             ):
        
#            frame = cv2.imread(x)
#            rows,cols=frame.shape
        
        img_json = []
        img_x_train = []
        img_y_train = []
        img_names = ""
        
        xxxx = np.array(self.X_DATASET)
        
        print("****") 
        print(xxxx.shape) 
        idx_owners=0
        idx_chars=0
        
        for owners in xxxx:
            print(idx_owners)
            print(self.OWNERS[idx_owners]) 
            
            for chars in owners:
                print(idx_chars)  #used
                print(self.CHARACTERS[idx_chars])
                
#                if (imgprocess=="real") and (showandsave==False) :
#                #train deep lerning here
#                    img_x_train.extend(chars)
#                    img_y_train.extend(to_categorical(np.ones((len(chars),), dtype=int)*idx_chars, num_classes = len(self.CHARACTERS)))
#                
                if imgprocess=="FisherFaces" :
                    tmpx=self.faceProcess1(
                                 characters=chars,
                                 owners=self.OWNERS[idx_owners],
                                 chars=self.CHARACTERS[idx_chars],
                                 numberoffaces=20,
                                 h_resize=h_resize,w_resize=w_resize,
                                 showandsave=showandsave
                                 )
                    img_x_train.extend(tmpx)
                    img_y_train.extend(to_categorical(np.ones((len(tmpx),), dtype=int)*idx_chars, num_classes = len(self.CHARACTERS)))
                    
                    
                
                img_names=str(img_names) + str(self.OWNERS[idx_owners]) + str(self.CHARACTERS[idx_chars])
                img_json.append([ idx_chars , str(self.OWNERS[idx_owners]) , str(self.CHARACTERS[idx_chars]) ])
                

#                if (not((imgprocess=="real") and (showandsave==False))) and (imgprocess!="FisherFaces") :
                if (imgprocess!="FisherFaces") :
                    
                    img_y_train.extend(to_categorical(np.ones((len(chars),), dtype=int)*idx_chars, num_classes = len(self.CHARACTERS)))
                    
                    idx=0
                    for pics in chars:
                        print(pics.shape)
                        img_x_train.append( self.faceProcess(
                                 pics=pics,
                                 owners=str(self.OWNERS[idx_owners]),
                                 chars=str(self.CHARACTERS[idx_chars]),
                                 idx=idx,
                                 numberoffaces=numberoffaces,
                                 h_resize=h_resize,w_resize=w_resize,
                                 imgprocess=imgprocess,
                                 showandsave=showandsave
                                 ) )
     
                        idx=idx+1 
                
    
                idx_chars=idx_chars+1
                    
            idx_owners=idx_owners+1 
            
        cv2.destroyAllWindows()
        
        y=np.array(img_y_train).astype('int')
        x=np.array(img_x_train).astype('float')
        print(x.shape)
#        print(len(x.shape))
        print(y.shape)
        
        return x,y,img_names,img_json
    
    
    
    def faceProcess(self,
                             pics=[],
                             owners="",
                             chars="",
                             idx=0,
                             numberoffaces=20,
                             h_resize=300,w_resize=300,
                             imgprocess="real",
                             showandsave=False
                             ):

        path=".\\IMGDEBUG"
        if(not os.path.exists(path)):
            os.mkdir(path)
            
        name=str(owners)+str(chars)+str(idx)
    
        owners_path=path+"\\"+name

#        res1=[]
#        res1.append(pics) 
#        res=res1[0]
        
        res = cv2.resize(pics,(w_resize, h_resize), interpolation = cv2.INTER_CUBIC)
#        res=pics
        
#        print(res)
        
        ## settings for LBP
        radius = 10
        n_points = 30
#        METHOD = 'uniform'
        
#        fig = plt.figure()
        if imgprocess=="Blurring" :
#            blur = cv2.blur(res,(3,3))
            blur = cv2.bilateralFilter(res,9,75,75)
            if showandsave :
                cv2.imshow('Blurring',blur)
                cv2.imwrite(owners_path+"_"+"blur"+".png",blur)   
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            return blur
    
    
    
    
        if imgprocess=="real" :
            if showandsave :
                cv2.imshow('roi_color',res)
                cv2.imwrite(owners_path+"_"+"res"+".png",res)   
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            return res

    
        resgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        
        
        if imgprocess=="gray" :
            if showandsave :
                cv2.imshow('gray',resgray)
                cv2.imwrite(owners_path+"_"+"resgray"+".png",resgray)   
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            return resgray
        
        
        if imgprocess=="local_binary_pattern" :            
            lbp = local_binary_pattern(resgray, n_points, radius,  method='default')
            if showandsave :
                cv2.imshow('lbp',lbp)
                cv2.imwrite(owners_path+"_"+"lbp"+".png",lbp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            return lbp


        if imgprocess=="Eigenfaces" : 
    #            similar to Eigenfaces 
            fig = plt.figure()
            ax1 = plt.subplot() 
            ax1.axis('off')
            imgplot1 = ax1.imshow(resgray)
            imgplot1.set_cmap('nipy_spectral')
            
            plot_img_np = self.get_img_from_fig(fig)
            h=plot_img_np.shape[0]
            w=plot_img_np.shape[1]
            for y in range(h) :
                yy=plot_img_np[y+1,int(w/2)]-plot_img_np[y,int(w/2)]
                if any(yy!=0) : break
            for x in range(w) :
                xx=plot_img_np[int(h/2),x+1]-plot_img_np[int(h/2),x]
                if any(xx!=0) : break
            y=y+5
            x=x+5
            h=int((plot_img_np.shape[0]-(y*2)))
            w=int((plot_img_np.shape[1]-(x*2)))
            img_np=plot_img_np[y:y+h,x:x+w]
            img_np1 = cv2.resize(img_np,(w_resize, h_resize), interpolation = cv2.INTER_CUBIC)
    #        print(img_np1.shape)
            if showandsave :
                cv2.imwrite(owners_path+"_"+"nipy_spectral"+".png",img_np1)
        #        fig.savefig(owners_path+"_"+"nipy_spectral"+".png")
                plt.close(fig)
                img1 = cv2.imread(owners_path+"_"+"nipy_spectral"+".png")
                cv2.imshow("nipy_spectral",img1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            return img_np1
        
        
        if imgprocess=="hsv" :
    #        similar to Eigenfaces       
            hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    #        hsvresgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            if showandsave :
                cv2.imshow('hsv',hsv)
                cv2.imwrite(owners_path+"_"+"hsv"+".png",hsv)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            return hsv
    
    
        if imgprocess=="edges" :
            edges = cv2.Canny(res,50,100)
            if showandsave :
                cv2.imshow('edges',edges)
                cv2.imwrite(owners_path+"_"+"edges"+".png",edges)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            return edges
    
        if imgprocess=="laplacian" :
            laplacian = cv2.Laplacian(res,cv2.CV_64F)
            if showandsave :
                cv2.imshow('laplacian',laplacian)
                cv2.imwrite(owners_path+"_"+"laplacian"+".png",laplacian)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            return laplacian
    
    
#        similar to LocalBinaryPatterns
        if imgprocess=="lbp0" :   
            sobelx = cv2.Sobel(resgray,cv2.CV_64F,1,0,ksize=5)
            if showandsave :
                cv2.imshow('sobelx',sobelx)
                cv2.imwrite(owners_path+"_"+"sobelx"+".png",sobelx)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            return sobelx
        
        
        if imgprocess=="lbp1" : 
            sobelx = cv2.Sobel(resgray,cv2.CV_64F,1,0,ksize=5)
            ret,thresh1 = cv2.threshold(sobelx,127,255,cv2.THRESH_TOZERO)
            if showandsave :
                cv2.imshow('thresh1',thresh1)
                cv2.imwrite(owners_path+"_"+"thresh1"+".png",thresh1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass 
            return thresh1
        
#        cv2.destroyAllWindows()
                            
    
    
        
    def faceProcess1(self,
                             characters=[],
                             owners="",
                             chars="",
                             numberoffaces=20,
                             h_resize=300,w_resize=300,
                             showandsave=False
                             ):
        character=[]
        character.extend(characters)
        
#        print(character)
        
        path=".\\IMGDEBUG"
        if(not os.path.exists(path)):
            os.mkdir(path)
 
        
        height, width = h_resize,w_resize
        nF  = numberoffaces  # number of pictures
        A = np.zeros([nF, height * width])
        Y = 0 #50 faces
        pca = PCA(n_components=4)
        B = np.zeros([nF, height , width])
        C = []
        
#        similar to FisherFaces
        idx=0
        for res in character:
            pics = cv2.resize(res,(w_resize, h_resize), interpolation = cv2.INTER_CUBIC)
#            pics=res
            A[Y, :]=color.rgb2gray(pics).reshape([1, height * width])
            if Y == (nF-1) :
                pass
            
                pca.fit(A[ : , : ])
                fig = plt.figure()
                for i in range(4):
                    ax = plt.subplot(2,2,i + 1)
                    ax.axis('off')
                    B[i ,:,:]=pca.components_[i].reshape([width, height])
#                    print(B[i ,:,:].shape)
                    C.append(B[i ,:,:].tolist())
                    ax.imshow(B[i ,:,:])

                if showandsave :
                    name=str(owners)+str(chars)+str(idx)
                    owners_path=path+"\\"+name     
    
                    fig.savefig(owners_path+"_"+str(i)+".png")
    #                    plt.close(fig)
                    img0 = cv2.imread(owners_path+"_"+str(i)+".png")
                    cv2.imshow("Eigenfaces"+str(i),img0)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        pass
                
                idx=idx+1
                Y=-1
            Y=Y+1   

#    cv2.destroyAllWindows()
            
        return C   
    
    
    
    def faceDetectionReadDebug(self,
                             numberoffaces=20,
                             h_resize=300,w_resize=300
                             ):
        
#            frame = cv2.imread(x)
#            rows,cols=frame.shape
        
        img_json = []
        img_x_train = []
        img_y_train = []
        img_names = ""
        
        xxxx = np.array(self.X_DATASET)
        
        print("****") 
        print(xxxx.shape) 
        idx_owners=0
        idx_chars=0
        
        for owners in xxxx:
            print(idx_owners)
            print(self.OWNERS[idx_owners]) 
            
            for chars in owners:
                print(idx_chars)  #used
                print(self.CHARACTERS[idx_chars])
                #train deep lerning here
                img_x_train.extend(chars)
                img_y_train.extend(to_categorical(np.ones((len(chars),), dtype=int)*idx_chars, num_classes = len(self.CHARACTERS)))
                img_names=str(img_names) + str(self.OWNERS[idx_owners]) + str(self.CHARACTERS[idx_chars])
                img_json.append([ idx_chars , str(self.OWNERS[idx_owners]) , str(self.CHARACTERS[idx_chars]) ])
                
                self.faceProcessDebug1(
                             characters=chars,
                             owners=self.OWNERS[idx_owners],
                             chars=self.CHARACTERS[idx_chars],
                             numberoffaces=20,
                             h_resize=300,w_resize=300
                             )
                
                idx=0
                for pics in chars:
                    print(pics.shape)
                    
                    self.faceProcessDebug(
                             pics=pics,
                             owners=str(self.OWNERS[idx_owners]),
                             chars=str(self.CHARACTERS[idx_chars]),
                             idx=idx,
                             numberoffaces=numberoffaces,
                             h_resize=h_resize,w_resize=w_resize
                             )
 
                    idx=idx+1 
                
    
                idx_chars=idx_chars+1
                    
            idx_owners=idx_owners+1 
            
        cv2.destroyAllWindows()
        
        y=np.array(img_y_train).astype('int')
        x=np.array(img_x_train).astype('float')
        print(x.shape)
        print(y.shape)
        

    # define a function which returns an image as numpy array from figure
    def get_img_from_fig(self,fig, dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        return img


    def faceProcessDebug(self,
                             pics=[],
                             owners="",
                             chars="",
                             idx=0,
                             numberoffaces=20,
                             h_resize=300,w_resize=300
                             ):

        path=".\\IMGDEBUG"
        if(not os.path.exists(path)):
            os.mkdir(path)
            
        name=str(owners)+str(chars)+str(idx)
    
        owners_path=path+"\\"+name

#        res1=[]
#        res1.append(pics) 
#        res=res1[0]
        
        res = cv2.resize(pics,(w_resize, h_resize), interpolation = cv2.INTER_CUBIC)
#        res=pics
        
#        print(res)
        
        ## settings for LBP
        radius = 10
        n_points = 30
#        METHOD = 'uniform'
        
        fig = plt.figure()

                       
        resgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(resgray, n_points, radius,  method='default')
        cv2.imshow('lbp',lbp)
        cv2.imwrite(owners_path+"_"+"lbp"+".png",lbp)


#            similar to Eigenfaces 
        ax1 = plt.subplot() 
        ax1.axis('off')
        imgplot1 = ax1.imshow(resgray)
        imgplot1.set_cmap('nipy_spectral')
        
        plot_img_np = self.get_img_from_fig(fig)
        h=plot_img_np.shape[0]
        w=plot_img_np.shape[1]
        for y in range(h) :
            yy=plot_img_np[y+1,int(w/2)]-plot_img_np[y,int(w/2)]
            if any(yy!=0) : break
        for x in range(w) :
            xx=plot_img_np[int(h/2),x+1]-plot_img_np[int(h/2),x]
            if any(xx!=0) : break
        y=y+5
        x=x+5
        h=int((plot_img_np.shape[0]-(y*2)))
        w=int((plot_img_np.shape[1]-(x*2)))
        img_np=plot_img_np[y:y+h,x:x+w]
        img_np1 = cv2.resize(img_np,(w_resize, h_resize), interpolation = cv2.INTER_CUBIC)
#        print(img_np1.shape)
        cv2.imwrite(owners_path+"_"+"nipy_spectral"+".png",img_np1)
#        fig.savefig(owners_path+"_"+"nipy_spectral"+".png")
        plt.close(fig)
        img1 = cv2.imread(owners_path+"_"+"nipy_spectral"+".png")
        cv2.imshow("nipy_spectral",img1)
        
        
#        similar to Eigenfaces       
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
#        hsvresgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        cv2.imshow('hsv',hsv)
        cv2.imwrite(owners_path+"_"+"hsv"+".png",hsv)
    
    
        edges = cv2.Canny(res,50,100)
        cv2.imshow('edges',edges)
        cv2.imwrite(owners_path+"_"+"edges"+".png",edges)
    

        laplacian = cv2.Laplacian(res,cv2.CV_64F)
        cv2.imshow('laplacian',laplacian)
        cv2.imwrite(owners_path+"_"+"laplacian"+".png",laplacian)
    
    
#        similar to LocalBinaryPatterns
        sobelx = cv2.Sobel(resgray,cv2.CV_64F,1,0,ksize=5)
        cv2.imshow('sobelx',sobelx)
        cv2.imwrite(owners_path+"_"+"sobelx"+".png",sobelx)
        
        ret,thresh1 = cv2.threshold(sobelx,127,255,cv2.THRESH_TOZERO)
        cv2.imshow('thresh1',thresh1)
        cv2.imwrite(owners_path+"_"+"thresh1"+".png",thresh1)
        
        
        cv2.imshow('roi_color',res)
        cv2.imwrite(owners_path+"_"+"res"+".png",res)     
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
        
#        cv2.destroyAllWindows()
                            
        
        
    def faceProcessDebug1(self,
                             characters=[],
                             owners="",
                             chars="",
                             numberoffaces=20,
                             h_resize=300,w_resize=300
                             ):
        character=[]
        character.extend(characters)
        
#        print(character)
        
        path=".\\IMGDEBUG"
        if(not os.path.exists(path)):
            os.mkdir(path)
 
        
        height, width = h_resize,w_resize
        nF  = numberoffaces  # number of pictures
        A = np.zeros([nF, height * width])
        Y = 0 #50 faces
        pca = PCA(n_components=4)
        B = np.zeros([nF, height , width])
        
#        similar to FisherFaces
        idx=0
        for res in character:
            pics = cv2.resize(res,(w_resize, h_resize), interpolation = cv2.INTER_CUBIC)
#            pics=res
            A[Y, :]=color.rgb2gray(pics).reshape([1, height * width])
            if Y == (nF-1) :
                pass
            
                pca.fit(A[ : , : ])
                fig = plt.figure()
                for i in range(4):
                    ax = plt.subplot(2,2,i + 1)
                    ax.axis('off')
                    B[i ,:,:]=pca.components_[i].reshape([width, height])
                    ax.imshow(B[i ,:,:])
                    
                name=str(owners)+str(chars)+str(idx)
                owners_path=path+"\\"+name     

                fig.savefig(owners_path+"_"+str(i)+".png")
#                    plt.close(fig)
                img0 = cv2.imread(owners_path+"_"+str(i)+".png")
                cv2.imshow("Eigenfaces"+str(i),img0)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
                
                idx=idx+1
                Y=-1
            Y=Y+1  

#    cv2.destroyAllWindows()
        
 
    
    def saveModelNameWithData(self , owners="" , chars="" , data=""):
        pass
        path=".\\IMGMODEL"
        if(not os.path.exists(path)):
            os.mkdir(path)       
        jsonpath=".\\IMGJSON"
        if(not os.path.exists(jsonpath)):
            os.mkdir(jsonpath)
            
        names = path+"\\"+str(owners)+"_"+str(chars)
        
        with open(jsonpath + "\\"+str(owners)+"_"+str(chars) + ".txt", 'w') as outfile:
            json.dump(data, outfile)
        
        return names 
    
    
    def saveInputWithData(self , name="" , imgprocess="" , data=""):
        pass
        path=".\\IMGMODEL"
        if(not os.path.exists(path)):
            os.mkdir(path)       
        jsonpath=".\\IMGINPUT"
        if(not os.path.exists(jsonpath)):
            os.mkdir(jsonpath)
            
        names = path+"\\"+str(name)+"_"+str(imgprocess)
        
        y = list(data)
        y[0] = str(imgprocess)
        data = tuple(y)
        
        with open(jsonpath + "\\"+str(name)+"_"+str(imgprocess) + ".txt", 'w') as outfile:
            json.dump(data, outfile)
        
        return names  
    
    
    def readModelWithData(self):
        path=".\\IMGMODEL"
        jsonpath=".\\IMGJSON"
        jsonpath1=".\\IMGINPUT"
        dirs = os.listdir(path)
#        print(dirs)
        self.__MODELS = []
        for models in dirs:
            pass
        
            if models.find(".hdf51") <= 0 : continue
        
            names=models.replace(".hdf51", "") 
            jsonnames=names.replace(path+"\\", "") 
            
            with open(jsonpath + "\\" + jsonnames + ".txt") as json_file:
                data = json.load(json_file)
                
#            print(data)  
            
            da=np.array(data)
            self.__MODELOWNERS = []
            self.__MODELCHARS = []
            self.__MODELLABEL = []

            
            for d in da :
                self.__MODELOWNERS.append(d[1])
                self.__MODELCHARS.append(d[2])
                self.__MODELLABEL.append(int(d[0]))
                
                
            self.MODELS.append(path + "\\" + names)
            self.MODELOWNERS.append(self.__MODELOWNERS)
            self.MODELCHARS.append(self.__MODELCHARS)
            self.MODELLABEL.append(self.__MODELLABEL)
            
            
            with open(jsonpath1 + "\\" + jsonnames + ".txt") as json_file:
                data1 = json.load(json_file)
            l=len(data1)    
#            print(l)
#            data1[0]
            self.MODELIMGPROCESS.append(data1[0])
            self.MODELSIZEX.append(data1[1])
            self.MODELSIZEY.append(data1[2])
            if l==4 :
                self.MODELSIZEZ.append(data1[3])
            else :
                self.MODELSIZEZ.append(1)
                
        return self.MODELS
    
    
    
    
#rcv=ReadCaptureVideo_class()
#rcv.readPNG()
##rcv.faceDetectionReadDebug(
##                         numberoffaces=20,
##                         h_resize=300,w_resize=300
##                         )
#
#rcv.faceDetectionRead(
#                         numberoffaces=20,
#                         h_resize=300,w_resize=300,
#                         #"real","FisherFaces","gray","local_binary_pattern","Eigenfaces","hsv","edges","laplacian","lbp0","lbp1"
#                         imgprocess="real",   
#                         showandsave=False
#                             )