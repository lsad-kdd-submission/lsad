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


dataName = 'multimnist'
batchSize = 32
learnRate = 0.001
positive_ratio = 0.003
maxEpochs=201
repeat = 5


P_X,U_X,N_list,N_list1,N_list2,A_list,testX,testY=data.load(dataName)
    
A_X = P_X[0:int(positive_ratio*len(N_list)) ]
U_shuffle_list = [x for x in range(len(U_X))]
N_shuffle_list = [x for x in range(len(N_list))]
A_shuffle_list = [x for x in range(len(A_X))]
dim_list = [784,128,64,32]

final_auc_history=[]
final_pr_history=[]
best_auc_history=[]

for rep in range(repeat):
    tf.reset_default_graph()

    with tf.Graph().as_default() as graph:

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config=config, graph=graph) as sess:
            
            
            model = models.DeepSAD("SAD",dim_list)
            model.build_optimizer(learning_rate = learnRate)
            sess.run(tf.global_variables_initializer())
            topk_len = tf.placeholder(tf.int32)
            
            #Auto-Encoder Pretraining. (option)
            '''
            for epoch in range(2000):    
                np.random.shuffle(A_shuffle_list)
                np.random.shuffle(U_shuffle_list)
                batch_U_list = np.random.choice( np.arange(len(U_shuffle_list)), batchSize)
                #train
                sess.run( model.pretrainStep, feed_dict={
                        model.neg_X: U_X[ batch_U_list ],
                        model.batchSize: int(batchSize)
                    })
            '''

            centroid = sess.run(model.avg_rep, feed_dict={ model.neg_X:U_X})
            
            for epoch in range(maxEpochs):
                np.random.shuffle(A_shuffle_list)
                np.random.shuffle(U_shuffle_list)
                
                trainSize= min( len(U_shuffle_list),len(A_shuffle_list) )
                batch_U_list = np.random.choice( np.arange(len(U_shuffle_list)), int(batchSize/2))
                batch_A_list = np.random.choice( np.arange(len(A_shuffle_list)), int(batchSize/2))
                
                #train
                sess.run( model.trainStep, feed_dict={
                        model.pos_X: A_X[ batch_A_list ],
                        model.neg_X: U_X[ batch_U_list ],
                        model.batchSize: int(batchSize/2),
                        model.center:centroid
                    })

            #Test
            test_predicted_label= []
            batch_vec=[]
            testSize=len(testY)
            batch_test_norm = sess.run( model.scores,
                                                         feed_dict={
                                                             model.neg_X: testX,
                                                             model.batchSize :len(testX),
                                                             model.center:centroid
                                                         })

            test_predicted_label.extend(batch_test_norm.tolist())
            test_AUC = roc_auc_score(testY,test_predicted_label)
            test_PR  = average_precision_score(testY,test_predicted_label)

            print("ROC:%s PR:%s"%(test_AUC,test_PR))
                    
    final_auc_history.append(test_AUC)
    final_pr_history.append(test_PR)
                
print("--------------------------------------------------")
print("DataName:%s Labeled_anomaly:%s batchSize:%s learningRate:%s DeepSAD" %(dataName,int(positive_ratio*len(N_list)),batchSize,learnRate))
print("ROC history %s"%(final_auc_history))
print("PR history %s"%(final_pr_history))
print("averaged ROC stddev averaged PR stddev")
print(np.mean(final_auc_history) , np.std(final_auc_history), np.mean(final_pr_history), np.std(final_pr_history) )