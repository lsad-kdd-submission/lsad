import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.contrib.opt import AdamWOptimizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, auc, roc_curve, roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score
import models
import data
import os
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import RadiusNeighborsClassifier
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import random_projection

dataName = 'multimnist'
batchSize = 32
learnRate = 0.001
positive_ratio = 0.003
maxEpochs=201
repeat = 5
hop = 3
K = 30

P_X,U_X,N_list,N_list1,N_list2,A_list,testX,testY=data.load(dataName)
    
A_X = P_X[0:int(positive_ratio*len(N_list)) ]
U_shuffle_list = [x for x in range(len(U_X))]
N_shuffle_list = [x for x in range(len(N_list))]
A_shuffle_list = [x for x in range(len(A_X))]
dim_list = [784,128,64,32]


final_auc_history=[]
final_pr_history=[]
best_auc_history=[]

trainX = np.concatenate((U_X,A_X),0)

for rep in range(repeat):
    tf.reset_default_graph()

    with tf.Graph().as_default() as graph:

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config=config, graph=graph) as sess:
            
            model = models.LSAD("LS",hop,dim_list,K)
            model.build_optimizer(learning_rate = learnRate)
            
            #TPA
            print("\nConverting a set of data points to simplicial complexes\n")
            if K <= 0:
                params={}
                params['TRAIN']=True
                params['K']=1
                train_indices = model.TPA(trainX,params,'persistent-homology')
                train_graph_tensor = tf.SparseTensorValue(train_indices, [1./(params['K'])]*len(train_indices),
                                       (len(trainX),len(trainX)))

                params['TRAIN']=False
                indices = model.TPA(testX,params,'persistent-homology')
                test_indices = np.concatenate( (train_indices,indices),0)
                test_graph_tensor=tf.SparseTensorValue(test_indices, [1./(params['K'])]*len(test_indices),
                                      (len(trainX)+len(testX),len(trainX)+len(testX)))
            else:
                params={}
                params['TRAIN']=True
                params['K']= K+1
                train_indices = model.TPA(trainX,params,'closest-K')
                train_graph_tensor = tf.SparseTensorValue(train_indices, [1./(params['K'])]*len(train_indices),
                                       (len(trainX),len(trainX)))
                
                params['TRAIN']=False
                indices = model.TPA(testX,params,'closest-K')
                test_indices = np.concatenate( (train_indices,indices),0)
                test_graph_tensor=tf.SparseTensorValue(test_indices, [1./(params['K'])]*len(test_indices),
                                      (len(trainX)+len(testX),len(trainX)+len(testX)))
                
                
            
            sess.run(tf.global_variables_initializer())
            for epoch in range(maxEpochs):
                print("Training: [%s/%s]"%(epoch+1,maxEpochs))

                np.random.shuffle(A_shuffle_list)
                np.random.shuffle(U_shuffle_list)
                
                trainSize= min( len(U_shuffle_list),len(A_shuffle_list) )
                
                batch_U_list = np.random.choice( np.arange(len(U_shuffle_list)), int(batchSize/2))
                batch_A_list = np.random.choice( np.arange(len(A_shuffle_list)), int(batchSize/2))
                
                #train
                sess.run(model.trainStep, feed_dict={
                    model.X : trainX,
                    model.A : train_graph_tensor,
                    model.P_Indices: len(U_X) + np.array(batch_A_list),
                    model.U_Indices: np.array(batch_U_list)
                    
                })
                
            #Test
            test_predicted_label=sess.run(model.scores,
                                    feed_dict={
                                        model.X: np.concatenate((trainX,testX),0),
                                        model.A: test_graph_tensor,
                                        model.U_Indices: len(trainX)+np.arange(len(testX))
                                    })


            test_AUC = roc_auc_score(testY,test_predicted_label)
            test_PR  = average_precision_score(testY,test_predicted_label)
            print("ROC:%s PR:%s"%(test_AUC,test_PR))
                    
    final_auc_history.append(test_AUC)
    final_pr_history.append(test_PR)
    
print("--------------------------------------------------")
print("DataName:%s Labeled_anomaly:%s batchSize:%s learningRate:%s DeepGPU K:%s hop:%s" %(dataName,int(positive_ratio*len(N_list)),batchSize,learnRate,K,hop))
print("ROC history %s"%(final_auc_history))
print("PR history %s"%(final_pr_history))
print("averaged ROC stddev averaged PR stddev")
print(np.mean(final_auc_history) , np.std(final_auc_history), np.mean(final_pr_history), np.std(final_pr_history) )