# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:48:32 2019


@author: Xiaojun Ding
"""




  
import numpy as np
import operator  
import heapq
import math

import time    
from sklearn import metrics    
import pickle as pickle    
import pandas as pd 

from sklearn import preprocessing


from sklearn.datasets import load_iris    
from sklearn import neighbors    
import sklearn    

from sklearn.model_selection import train_test_split


from sklearn.metrics import classification_report

from sklearn import svm  # svm支持向量机

import pandas as pd
import urllib

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


from sklearn.svm import SVR


from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import Ridge




from collections import Counter


import datetime


import warnings

warnings.filterwarnings("ignore")


class IWKNN:        
    
    def __init__(self,K=-1,n_iter=100):          
        self.K=K
        self.n_iter=n_iter        
     

    def computeKNNDistanceAndDerivative(self,i_th,X,Y,w):           
   
        
        x_i=X[i_th]
        
        nX = X.shape[0] # shape[0] stands for the num of row 
         
         
        diff = np.tile(x_i, (nX, 1)) - X # Subtract element-wise 
        dw_dist=diff*diff               # multiply element-wise
        dist=np.dot(dw_dist,w)         # weighted distance
        dist=np.sqrt(dist)
        
        
        sortedDistIndices = np.argsort(dist,axis=0 )       
            
        dw=np.zeros((x_i.shape[0],))
        
        
        borderFlag=0
        
        for neighbour in range(1,self.K+1):    # do not include itself
            
            k=sortedDistIndices[neighbour]     
            
            if   Y[i_th]!=Y[k]:
                borderFlag=1
            
        if borderFlag==0:
            return [dw,borderFlag]            #  derivative=0 when it is not a border point
        
        
       
        for neighbour in range(1,self.K+1):    # do not include itself
            
            k=sortedDistIndices[neighbour]        
            
            product_elementwise=(x_i-X[k])*(x_i-X[k])
            
            dist_k=np.sqrt( np.dot( product_elementwise, w) )+0.000001  # avoid dist_k=0
            
            dw_k=1/2*1/dist_k*product_elementwise       
        
            if   Y[i_th]!=Y[k]:   
                dw-=dw_k           
            else:       
                dw+=dw_k
        
        return  [dw,borderFlag]
        
    
        
   
        
        
        
    def fit( self,X, Y):    
        
        print(" ---------------------fit-------------------")   
        
        
        # w=np.random.random((X.shape[1]))      
        
        w=np.ones((X.shape[1]))
        
        minError=100000
        
        
        if self.K==-1:
           [K, errorCount]= self.findMinErrorK(X,Y,w)
           self.K=K 
           minError=errorCount
        
        bestK=self.K
        
        for i in range(10):            
            initialW=np.random.random((X.shape[1])) 
            [K, errorCount]= self.findMinErrorK(X,Y,initialW,bestK-1,bestK)
            if errorCount<minError:
                minError=errorCount
                w=initialW
                
        self.K=bestK
            
                     
        n_iter=self.n_iter
        
        
        threshold=np.sqrt(1.0/len(w))*0.01
        
        borderPoints=[]
        
        for t in range(0,n_iter):      
            
            nX = X.shape[0] # shape[0] stands for the num of row 
            
            dw=np.zeros(w.shape)#       
            
            if t%10==0:   
                
                borderPoints=[]
                
                for i in range(nX):               
             
                    [dw_i,borderFlag]=self.computeKNNDistanceAndDerivative(i,X, Y, w)                    
                    dw+=dw_i       
                    
                   
                    
                    if borderFlag==1:
                        borderPoints.append(i)
            else:
                
                 for i in borderPoints:   
                     
                     [dw_i,borderFlag]=self.computeKNNDistanceAndDerivative(i,X, Y, w)                    
                     dw+=dw_i     
                     
            # w=[1,2,3,4,5]
            # threshold=1
            # dd=(np.random.random(len(w))-0.5*np.ones(len(w)))*threshold   
            
            dw+=np.random.random(len(w))*0.5*threshold   
                
            dw=dw/(np.dot(dw,dw)**0.5+0.000000001)
           
            learning_rate=(n_iter-t)*0.1/n_iter+ 0.01
            
             
            w=w-learning_rate*dw         
            
            
            for n in range(w.shape[0]):            
                if w[n]<threshold:
                    w[n]=0        
            
            w=w/(np.dot(w,w)**0.5+0.000000001)       
            
#            print ("epoch =",t)
#            print(" dw=  ", w )       
            


        
           
        
        threshold=np.sqrt(1.0/len(w))*0.01
        
        for i in range(w.shape[0]):
            if(w[i]<threshold):
                w[i]=0
                
        w=w/(np.dot(w,w)**0.5+0.000000001)   
         
        return w  
    
    
    def predict_one(self, new_x, X, Y,w):     
            
           
        nX = X.shape[0] # shape[0] stands for the num of row 
         
         
        diff = np.tile(new_x, (nX, 1)) - X # Subtract element-wise 
        dw_dist=diff*diff               # multiply element-wise
        dist=np.dot(dw_dist,w)         # weighted distance
        dist=np.sqrt(dist)
        
        sortedDistIndices = np.argsort(dist,axis=0 )       
        
        classCount = {} # define a dictionary (can be append element)  
        for i in range(self.K):  
          
            voteLabel = Y[sortedDistIndices[i]]  
      
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
      
      
        maxCount = 0  
        for key, value in classCount.items():  
            if value > maxCount:  
                maxCount = value  
                maxIndex = key  
      
        return maxIndex   
    
    
    
    def predict(self, predict_x, X, Y, w):     
        
#        print(" ---------------------predict-------------------")
        
        
        if type(predict_x[0])==np.float64 or type(predict_x[0])==np.int64:
            return self.predict_one(predict_x,X,Y,w)    
        else:              
            predict_y=np.zeros((predict_x.shape[0],))
            
            for i in range(predict_x.shape[0]):
                predict_y[i]=self.predict_one(predict_x[i],X,Y,w)      
          
        return predict_y   
    
    
    
    
    
    
    def ErrorCount(self,X,Y,w):      
        
        count=0
        
        nF=X.shape[1]      
        
        px=np.ones((nF,))*100000  
        
        nX = X.shape[0] # shape[0] stands for the num of row 
        
        for i in range(nX):     
            
            predict_x=X[i]
            
            newX=X.copy()
            
            newX[i]=px            
            
            predict_y=self.predict(predict_x,newX,Y,w)
            
            if predict_y != Y[i]:
                
                count+=1          
                
#            print(i,predict_y,Y[i])
        
        return count
    
    
    def findMinErrorK(self,X,Y, w, minK=1,maxK=15):
        
        bc=np.bincount(Y)
        
        minC=min(bc)
        
        maxK=max(1,min(minC-1,maxK))
        
        minErrorK=X.shape[0]
        minErrorCount=X.shape[0]
        
        for i in range(maxK,minK-1,-2):
            
            self.K=i
            c=self.ErrorCount(X,Y,w)
            if c<minErrorCount:
                minErrorCount=c
                minErrorK=self.K
                
#            print('k=',i,'error=',c)
                
            
#        print('min k=',minErrorK,' min error=',minErrorCount)
        
        return [minErrorK,minErrorCount]
            
        
        
        
    
    def w_count(self,w):
    
        count=0
        
        threshold=np.sqrt(1.0/len(w))*0.01
        
        for i in range(w.shape[0]):
            if(w[i]>threshold):
                count+=1
           
        print('# features='+str(len(w))+' #selected features ='+str(count)+ ' ratio='+str(count/len(w)))
              
        return count











def ParsePerformanceStr(performanceStr):
    
    
    items=performanceStr.split( )

    precision=float(items[len(items)-4])
    recall=float(items[len(items)-3])
    fscore=float(items[len(items)-2])
    
    return [precision, recall, fscore]




def RecordPerformance(performance,y,y_predict):    
    
    performanceStr=classification_report(y, y_predict)
        
    p=ParsePerformanceStr(performanceStr)
    
    performance+=p    
   
    print(performanceStr)    
    
    
    

def TestPerformance(X,Y, testTimes=10, bScaled=1, _test_size=0.1):
    
    
    if bScaled==1:
        X_scaled = preprocessing.scale(X)        
        X=X_scaled
        
      
    print('--------------------START-----------')
    
    performances=np.zeros((4,3))
    
    ratio=0
  
    
    for iter in range(testTimes):       
    
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=_test_size)        
        
        model = IWKNN( )
        
        w=model.fit(X_train, y_train)       
       
        
        ratio+=model.w_count(w)/len(w)
        
        print('IWKNN w=',w)
        
        predict_y=model.predict(X_test,X_train,y_train,w)    
              
        
        print(" ------------------weighted KNN----------------")
        
        RecordPerformance(performances[0],y_test, predict_y) 
        
        
        #------------------------------------------------------
        
        nF=X_train.shape[1]
        
        originalW=np.ones((nF,))
        
        predict_y2=model.predict(X_test,X_train,y_train,originalW)        
      
        
        
        print(" ------------------original KNN----------------")

        
        RecordPerformance(performances[1],y_test, predict_y2) 


       #---------------------------
        clf = svm.SVC(kernel='linear')  #线性核函数        
        clf.fit(X=X_train, y=y_train)  # 训练模型。参数sample_weight为每个样本设置权重。应对非均衡问题
        svm_y = clf.predict(X_test)  # 使用模型预测值
        print(" ------------------Linear SVM ----------------")
        
#        print(clf._get_coef())
        RecordPerformance(performances[2],y_test, svm_y) 



       #---------------------------
        clf = svm.SVC(kernel='rbf')  #rbf核函数        
        clf.fit(X=X_train, y=y_train)  # 训练模型。参数sample_weight为每个样本设置权重。应对非均衡问题
        svm_y = clf.predict(X_test)  # 使用模型预测值
        print(" ------------------rbf SVM ----------------")
        
#        print(clf._get_coef())
        RecordPerformance(performances[3],y_test, svm_y) 


    ratio/=testTimes
    performances/=testTimes
    
    print('pricision  recall f-score')
    print(performances)
    
    print('#features=', X.shape[1], ' average ratio ', ratio)





        
      

    



def HitFeatures(hitResult, w,k,option='absoluteV'):
    
    if option=='absoluteV':
       
        absW=np.abs(w)
        
        idx=np.argsort(absW)[::-1]
        
        idxK=idx[0:k]
        
        for i in range(k):
            hitResult.append(idxK[i])
            
            
        d2=Counter(hitResult)
            
        sorted_x = sorted(d2.items(), key=lambda x: x[1], reverse=True)
 
        print(sorted_x)

            
        
        # print(Counter(hitResult))




def support2value(support):
    
    value=np.zeros(len(support))
   
    for i in range(len(support)):
        
        if( support[i]==True):
            value[i]=1
            
    return value



def TestPerformance2(X,Y, nF=3, testTimes=10, bScaled=1, _test_size=0.1):
    
    
    if bScaled==1:
        X_scaled = preprocessing.scale(X)        
        X=X_scaled
        
      
    print('--------------------START-----------')
        
  
    
    hitResult0=[]
    hitResult1=[]
    hitResult2=[]
    hitResult3=[]
    hitResult4=[]
    
    times=np.zeros(5,)
    
    for iter in range(testTimes):       
    
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=_test_size)     
        
        
        
        
        
        
        starttime = datetime.datetime.now()
        
        model = IWKNN( )
        
        w=model.fit(X_train, y_train)
        
        endtime = datetime.datetime.now()
        
        times[0]=times[0]+(endtime - starttime).seconds
        
             
        print('------------IWKNN------------')
        
        HitFeatures(hitResult0,w,nF)
        
        
        
        
        
        
        # fit an Extra Trees model to the data
        
        starttime = datetime.datetime.now()
        
        model = ExtraTreesClassifier()
        model.fit(X, Y)
        
        endtime = datetime.datetime.now()
        
        times[1]=times[1]+(endtime - starttime).seconds
        
        print('------------ExtraTreesClassifier------------')
        # display the relative importance of each attribute
        # print(model.feature_importances_)
        
        HitFeatures(hitResult1,model.feature_importances_,nF)
    
    
    
    
    
    
        starttime = datetime.datetime.now()
        model = LogisticRegression()
        # create the RFE model and select 3 attributes
        rfe = RFE(model, nF)
        rfe = rfe.fit(X, Y)
        
        endtime = datetime.datetime.now()
        
        times[2]=times[2]+(endtime - starttime).seconds
        
        # summarize the selection of the attributes
        print('------------rfe logistic regression------------')
        # print(rfe.support_)
        # print(rfe.ranking_)
        
        # print(support2value(rfe.support_))
        
        HitFeatures(hitResult2,support2value(rfe.support_),nF)
    
        #    print(rfe.scores_)
            
     
        
     
        starttime = datetime.datetime.now()
       
        model = svm.SVC(kernel='linear')
        # create the RFE model and select 3 attributes
        rfe = RFE(model, nF)
        rfe = rfe.fit(X, Y)
        
        endtime = datetime.datetime.now()
        
        times[3]=times[3]+(endtime - starttime).seconds
        
        # summarize the selection of the attributes
        print('------------rfe svm linear------------')
        # print(rfe.support_)
        # print(rfe.ranking_)
        
        # print(support2value(rfe.support_))
        
        HitFeatures(hitResult3,support2value(rfe.support_),nF)
        
        
        
        
        
        starttime = datetime.datetime.now()
        
        ridge = Ridge(alpha=1)
        ridge.fit(X, Y)
        
        
        endtime = datetime.datetime.now()
        
        times[4]=times[4]+(endtime - starttime).seconds
        
        print('------------ridge------------')
        # print (ridge.coef_)
        # print (ridge.intercept_)
        
        HitFeatures(hitResult4,ridge.coef_,nF)
        
        
        print('time=')
        print(times)
       
        
  

   




      
        
        
        



def RFE_Method(X,Y,nF):        
        
    
           
     # fit an Extra Trees model to the data
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print('------------ExtraTreesClassifier------------')
    # display the relative importance of each attribute
    print(model.feature_importances_)



    model = LogisticRegression()
    # create the RFE model and select 3 attributes
    rfe = RFE(model, nF)
    rfe = rfe.fit(X, Y)
    # summarize the selection of the attributes
    print('------------rfe logistic regression------------')
    print(rfe.support_)
    print(rfe.ranking_)

#    print(rfe.scores_)
        
 
   
    model = svm.SVC(kernel='linear')
    # create the RFE model and select 3 attributes
    rfe = RFE(model, nF)
    rfe = rfe.fit(X, Y)
    # summarize the selection of the attributes
    print('------------rfe svm linear------------')
    print(rfe.support_)
    print(rfe.ranking_)
    
    
    ridge = Ridge(alpha=1)
    ridge.fit(X, Y)
    print('------------ridge------------')
    print (ridge.coef_)
    print (ridge.intercept_)
  
#    model = svm.SVC(kernel="rbf") #rbf核函数  
#    # create the RFE model and select 3 attributes
#    rfe = RFE(model, nF)
#    rfe = rfe.fit(X, Y)
#    # summarize the selection of the attributes
#    print('------------rfe svm rbf------------')
#    print(rfe.support_)
#    print(rfe.ranking_)
#
#
#    model = SVR(kernel="rbf") #rbf核函数  
#    # create the RFE model and select 3 attributes
#    rfe = RFE(model, nF)
#    rfe = rfe.fit(X, Y)
#    # summarize the selection of the attributes
#    print('------------rfe svm rbf------------')
#    print(rfe.support_)
#    print(rfe.ranking_)



