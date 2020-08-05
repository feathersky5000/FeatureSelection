# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:23:55 2020

@author: 77994
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

from sklearn import datasets

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

from sklearn.datasets import make_blobs

import warnings

warnings.filterwarnings("ignore")











def LoadData(sample_str):
    
    
#---------------- simulated dataset 1-------------------
# there are 2 class 
# feature 0 decides which class the sample belongs to
    
    if sample_str == 's1':
         
         n=300

         X=np.random.random((n,100))
         Y=np.zeros(n)
        
         for i in range(n):
            if i<n/3:
                X[i][0]=-1+np.random.random()
                Y[i]=0
            else:
                X[i][0]=1+np.random.random()
                Y[i]=1
    

   
    

#---------------- simulated dataset 2-------------------
# feature 0, feature 2 and feature 4  decide which class the sample belongs to
# there are 3 classes
# in this dataset, the iwknn can achieve the best performnance than KNN and SVM
               
    if sample_str == 's2':
         
        n=300

        X=np.random.random((n,100))
        Y=np.zeros(n)
        
        for i in range(n):
            if i<n/3:
                X[i][0]=1+np.random.random()
                X[i][2]=0+np.random.random()
                X[i][4]=0+np.random.random()
                Y[i]=0
            elif i<2*n/3:
                X[i][0]=0+np.random.random()
                X[i][2]=1+np.random.random()
                X[i][4]=0+np.random.random()
                Y[i]=1
            else:
                X[i][0]=0+np.random.random()
                X[i][2]=0+np.random.random()
                X[i][4]=1+np.random.random()
                Y[i]=2
   
    
#---------------- simulated dataset 3-------------------
# feature 0, feature 19   decide which class the sample belongs to
# feature 0 and feature 19 have a interacitve effect. In class 1, there are two patterns [feature0=0, feature119=2] or [f0=2, f19=0]. In class 2, [f0=0, f19=0] 
# or  [f0=2, f19=2]. Only looking at feature 0, there is no difference between class 1 with class 2. But regarding f0 and f2 together, there is a great difference.

# there are 2 classes
# in this dataset, the iwknn can achieve the best performnance than KNN and SVM
                
    if sample_str == 's3':
         
        n=300

        X=np.random.random((n,100))
        Y=np.zeros(n)
        
        for i in range(n):
            if i<n/3:
                if i%2==0:
                    X[i][0]=0+np.random.random()
                    X[i][99]=2+np.random.random()
                else:
                    X[i][0]=2+np.random.random()
                    X[i][99]=0+np.random.random()
                Y[i]=0
            else:
                if i%2==0:
                    X[i][0]=2+np.random.random()
                    X[i][99]=2+np.random.random()
                else:
                    X[i][0]=0+np.random.random()
                    X[i][99]=0+np.random.random()       
                Y[i]=1






#---------------- simulated dataset 4-------------------
# feature 0, feature 2 and feature 4  decide which class the sample belongs to
# there are 3 classes
# in this dataset, the iwknn can achieve the best performnance than KNN and SVM
               
    if sample_str == 's4':
         
        n=400

        X=np.random.random((n,20))
        Y=np.zeros(n)
        
        for i in range(n):
            if i<n/4:
                X[i][0]=0+np.random.random()
                X[i][2]=0+np.random.random()             
                Y[i]=0
            elif i<2*n/4:
                X[i][0]=0+np.random.random()
                X[i][2]=1+np.random.random()              
                Y[i]=1
            elif i<3*n/4:
                X[i][0]=1+np.random.random()
                X[i][2]=0+np.random.random()              
                Y[i]=2                
            else:
                X[i][0]=1+np.random.random()
                X[i][2]=1+np.random.random()              
                Y[i]=3
                
                
                
                
                

#---------------- simulated dataset 5-------------------
# feature 0, feature 2 and feature 5  decide which class the sample belongs to
# there are 3 classes
# in this dataset, the iwknn can achieve the best performnance than KNN and SVM
               
    if sample_str == 's5':
         
        n=800

        X=np.random.random((n,800))
        Y=np.zeros(n)
        
        for i in range(n):
            if i<n/3:
                X[i][0]=1+np.random.random()
                X[i][2]=0+np.random.random()
                X[i][4]=0+np.random.random()
                Y[i]=0
            elif i<2*n/3:
                X[i][0]=0+np.random.random()
                X[i][2]=1+np.random.random()
                X[i][4]=0+np.random.random()
                Y[i]=1
            else:
                X[i][0]=0+np.random.random()
                X[i][2]=0+np.random.random()
                X[i][4]=1+np.random.random()
                Y[i]=2
                
                
                
           
                

#---------------- simulated dataset 6-------------------
# feature 0, feature 2 and feature 5  decide which class the sample belongs to
# there are 3 classes
# in this dataset, the iwknn can achieve the best performnance than KNN and SVM
               
    if sample_str == 's6':
         
        n=1000
        
        
        X0, Y0 = make_blobs(n_samples=n, n_features=5, centers=15,random_state=3)

        X=np.random.random((n,50))
        Y=np.zeros(n)
        
        X[:,0:5]=X0
        
        Y=np.array(Y0/3,dtype=int)
                                
                
      
        
      
            
#---------------- simulated dataset 3-------------------
# feature 0, feature 19   decide which class the sample belongs to
# feature 0 and feature 19 have a interacitve effect. In class 1, there are two patterns [feature0=0, feature119=2] or [f0=2, f19=0]. In class 2, [f0=0, f19=0] 
# or  [f0=2, f19=2]. Only looking at feature 0, there is no difference between class 1 with class 2. But regarding f0 and f2 together, there is a great difference.

# there are 2 classes
# in this dataset, the iwknn can achieve the best performnance than KNN and SVM
                
    if sample_str == 's7':
         
        n=300

        X=np.random.random((n,200))
        Y=np.zeros(n)
        
        for i in range(n):
            if i<n/3:
                if i%2==0:
                    X[i][0]=0+np.random.random()
                    X[i][19]=2+np.random.random()
                else:
                    X[i][0]=2+np.random.random()
                    X[i][19]=0+np.random.random()
                Y[i]=0
            else:
                if i%2==0:
                    X[i][0]=2+np.random.random()
                    X[i][19]=2+np.random.random()
                else:
                    X[i][0]=0+np.random.random()
                    X[i][19]=0+np.random.random()       
                Y[i]=1



                

    if sample_str == 'Iris':
        sample = datasets.load_iris()
        X = sample.data
        Y = sample.target
        
    if sample_str == 'Wine':
        sample = datasets.load_wine()
        X = sample.data
        Y = sample.target       
        

        
    if (sample_str == 'winequality-red') or (sample_str == 'winequality-white'):
                
        columns = ["facidity", "vacidity", "citric", "sugar", "chlorides", "fsulfur", 
                       "tsulfur", "density", "pH", "sulphates", "alcohol", "quality"]
        
#        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
#        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        
        wines=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"+sample_str+".csv",
                             names=columns, sep=";", skiprows=1)
        
   
       
        wines=np.array(wines,dtype=float)
        X=wines[:,0:11]
        Y=wines[:,11]
        Y=np.array(Y,dtype=int)     
        
        
    if sample_str=='wdbc':

        try:
            df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'
                             '/breast-cancer-wisconsin/wdbc.data', header=None)
        
        except urllib.error.URLError:
            df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                             'python-machine-learning-book/master/code/'
                             'datasets/wdbc/wdbc.data', header=None)
            
        print('rows, columns:', df.shape)
        df.head()
        
        
        from sklearn.preprocessing import LabelEncoder
        
        X = df.loc[:, 2:].values
        Y = df.loc[:, 1].values
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        le.transform(['M', 'B']) # array([1, 0])



    if sample_str=='TCGA-PANCAN-HiSeq':
        
               
        df_x = pd.read_csv('dataset/TCGA-PANCAN-HiSeq/data.csv')
        
        df_y = pd.read_csv('dataset/TCGA-PANCAN-HiSeq/labels.csv')        
       
        cX=df_x.loc[:,df_x.columns !=df_x.columns[0]]           
        
        cY=preprocessing.LabelEncoder().fit_transform(df_y['Class'])
        
        X=np.array(cX)
        Y=np.array(cY,dtype=int)      

                 

    X=np.array(X)
    Y=np.array(Y,int)        
    Y-=min(Y)
    print('nX=',X.shape[0],' nF=',X.shape[1],' k=',len(set(Y)))     
    print('cluster size=',np.bincount(Y))




    return [X,Y]
    




# compare the performances of classification algorithms

def TestDataset(dataset,testTimes=10):
    
    print(dataset)
    
    [X,Y]=LoadData(dataset)
    
    TestPerformance(X,Y,testTimes)
    
  
    


# compare the performances of feature selection algorithms

def TestDataset2(dataset,nF=3,testTimes=10):
    
    print(dataset)
    
    [X,Y]=LoadData(dataset)
    
    TestPerformance2(X,Y,nF=nF,testTimes=testTimes)
    
 
    




        
TestDataset("s1")     

TestDataset("s2")

TestDataset("s3")




TestDataset("Iris")                                                                                                                                     

TestDataset("Wine")   #UCI Machine learning database

TestDataset("wdbc")


TestDataset("TCGA-PANCAN-HiSeq")

TestDataset("TCGA-PANCAN-HiSeq",3)



TestDataset2("s1",1)

TestDataset2("s2",3)

TestDataset2("s3",2)

TestDataset2("s6",5)

TestDataset2("s4")


TestDataset2("s5",20)


TestDataset2("s7",2)


TestDataset2("wdbc",7)





