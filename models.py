import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import AdamWOptimizer
from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import radius_neighbors_graph
from scipy.spatial.distance import euclidean
import os
from ripser import ripser
from persim import plot_diagrams
        
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
class DeepSAD(object):
    
    def __init__(self,name,dim_list):

        assert len(dim_list) >= 2, 'specify input output dimension'
        self.name = name
        self.dim_list = dim_list
        with tf.variable_scope(self.name):
            self.center = tf.placeholder(tf.float32, name="center")
            self.anc_X = tf.placeholder(tf.float32, [None, dim_list[0] ])
            self.pos_X = tf.placeholder(tf.float32, [None, dim_list[0] ])
            self.neg_X = tf.placeholder(tf.float32, [None, dim_list[0] ])
            
            self.seq_len = tf.placeholder(tf.int32, [None])
            
            self.expanded_anc_X = tf.expand_dims(self.anc_X,2)
            self.expanded_pos_X = tf.expand_dims(self.pos_X,2)
            self.expanded_neg_X = tf.expand_dims(self.neg_X,2)
            
            self.batchSize = tf.placeholder(tf.int32)
            self.Nsamples = tf.placeholder(tf.int32)
            self.weight = tf.placeholder(tf.float32, [None,None])
            self.pos_layers = []
            self.anc_layers = []
            self.neg_layers = []
            
            self.dec_layers = []
         
           
            self.pos_layers.append(self.pos_X)
            self.anc_layers.append(self.anc_X)
            self.neg_layers.append(self.neg_X)
            with tf.variable_scope("MLP_Configuration"):
                for idx,dim in enumerate(dim_list):
                    if idx==0:continue
                    W = tf.get_variable('%s'%(idx) ,shape=[dim_list[idx-1],dim],initializer=tf.contrib.layers.xavier_initializer())
                    anc_layer = tf.matmul(self.anc_layers[idx-1],W)
                    pos_layer = tf.matmul(self.pos_layers[idx-1],W)
                    neg_layer = tf.matmul(self.neg_layers[idx-1],W)
                        
                    
                    if idx != len(dim_list)-1 and idx >= 1:
                        anc_layer = tf.nn.relu(anc_layer)
                        pos_layer = tf.nn.relu(pos_layer)
                        neg_layer = tf.nn.relu(neg_layer)
                    
                    self.anc_layers.append(anc_layer)
                    self.pos_layers.append(pos_layer)
                    self.neg_layers.append(neg_layer)
                    
                    
            self.dec_layers.append(self.neg_layers[-1])
            with tf.variable_scope("DECODER_Configuration"):
                idx = 0
                n = len(dim_list)
                for dim in reversed(dim_list):
                    idx+=1
                    if idx== n: break
                    W = tf.get_variable('dec%s'%(idx), shape=[dim, dim_list[n-1-idx]],initializer=tf.contrib.layers.xavier_initializer())
                    
                    dec_layer = tf.matmul(self.dec_layers[idx-1],W)
                    
                    bias= tf.get_variable('decb%s'%(idx) ,shape=[dim_list[n-1-idx]],initializer=tf.contrib.layers.xavier_initializer())
                    bias= tf.Variable(tf.zeros([dim_list[n-1-idx]]))
                    dec_layer = tf.nn.bias_add(dec_layer,bias)
                    
                    
                    if idx < n-1:
                        dec_layer = tf.nn.relu(dec_layer)
                    
                    self.dec_layers.append(dec_layer)
            
            self.dec_vec = self.dec_layers[-1]
            
            self.anc_vec = self.anc_layers[-1]
            self.pos_vec = self.pos_layers[-1]
            self.neg_vec = self.neg_layers[-1]
            self.avg_rep = tf.reduce_mean(self.neg_vec,axis=0)

            self.P = self.pos_vec - self.center
            self.U = self.neg_vec - self.center

            self.pos_dist = tf.reduce_sum(tf.square(self.pos_vec - self.center),axis=1,keep_dims=True)
            self.neg_dist = tf.reduce_sum(tf.square(self.neg_vec - self.center),axis=1,keep_dims=True)
            
            self.scores = self.neg_dist

    def build_optimizer(self,learning_rate):

        self.AEloss = tf.reduce_mean(tf.reduce_sum(tf.square(self.dec_vec-self.neg_X),axis=1))
        self.Ploss = tf.reduce_mean(tf.div(1.,self.pos_dist+0.000001))
        self.Uloss = tf.reduce_mean(self.neg_dist)
        self.loss = tf.reduce_mean(tf.div(1.,self.pos_dist+0.000001) + self.neg_dist)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.pretrainStep = self.optimizer.minimize(self.AEloss)
        self.trainStep = self.optimizer.minimize(self.loss)
          
class LSAD(object):
    
    def __init__(self,name,hop,dim_list,K):

        assert len(dim_list) >= 2, 'specify input output dimension'
        self.name = name
        self.dim_list = dim_list
        self.data = None
        self.neigh = None
        self.r = None
        with tf.variable_scope(self.name):
            self.center = tf.placeholder(tf.float32, name="center")
            self.X = tf.placeholder(tf.float32,   [None,dim_list[0] ])
            self.expanded_X = tf.expand_dims(self.X,2)
            #print(self.expanded_X.get_shape())
            self.A = tf.sparse_placeholder(tf.float32)
            self.P_Indices = tf.placeholder(tf.int32, [None])
            self.U_Indices = tf.placeholder(tf.int32, [None])
            
            self.Ws =[]
            self.layers = []

            

            self.layers.append(self.X)
            with tf.variable_scope("MLP_Configuration"):
                for idx,dim in enumerate(dim_list):
                    if idx==0: continue
                    W = tf.get_variable('%s'%(idx) ,shape=[dim_list[idx-1],dim],initializer=tf.contrib.layers.xavier_initializer())
                    layer = tf.matmul(self.layers[idx-1],W)
                        
                    bias= tf.get_variable('b%s'%(idx) ,shape=[dim],initializer=tf.contrib.layers.xavier_initializer())
                    bias= tf.Variable(tf.zeros([dim]))
                    layer = tf.nn.bias_add(layer,bias)
                    if idx < len(dim_list)-1 and idx >= 1 :
                        layer = tf.nn.relu(layer)

                    self.Ws.append(tf.reduce_sum(tf.square(W)))
                    self.layers.append(layer)

            
            self.vectors = [tf.nn.l2_normalize(self.layers[-1],axis=1)]    
            
            if K <=0:
                for i in range(hop):
                    self.vectors.append(tf.sparse_tensor_dense_matmul(self.A,self.vectors[i]))
                for i in range(1,hop+1):
                    self.vectors[i] = tf.nn.l2_normalize(self.vectors[i],axis=1)
            else:
                for i in range(hop):
                    self.vectors.append(tf.sparse_tensor_dense_matmul(self.A,self.vectors[i]))
                # (optional)
#                 for i in range(1,hop+1):
#                     self.vectors[i] = tf.nn.l2_normalize(self.vectors[i],axis=1)
                    
            self.P = []
            self.U = []
            for i in range(hop+1):
                self.P.append( tf.nn.embedding_lookup(self.vectors[i], self.P_Indices) )
                self.U.append( tf.nn.embedding_lookup(self.vectors[i], self.U_Indices) )
          
            self.expanded_P = []
            self.expanded_U = []
            for i in range(hop+1):
                self.expanded_P.append( tf.expand_dims(self.P[i],1))
                self.expanded_U.append( tf.expand_dims(self.U[i],1))
            
            assert hop >= 1, "hop is less then 1"
             
            self.P_ref = self.expanded_P[1]
            self.U_ref = self.expanded_U[1]
            for h in range(2,hop+1):
                self.P_ref = tf.concat((self.P_ref,self.expanded_P[h]),1)
                self.U_ref = tf.concat((self.U_ref,self.expanded_U[h]),1)
                
            
            self.P_loss = tf.reduce_mean( ( tf.reduce_mean( (tf.reduce_sum(self.expanded_P[0]*self.P_ref,2)),1) ) )
            
            self.U_loss = tf.reduce_mean( -tf.reduce_mean( (tf.reduce_sum(self.expanded_U[0]*self.U_ref,2)),1))
            self.scores = -tf.reduce_mean(tf.reduce_sum(self.expanded_U[0]*self.U_ref,2),1)#1d
    

    def build_optimizer(self,learning_rate):
        
        self.loss = self.P_loss + self.U_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.trainStep = self.optimizer.minimize(self.loss)
    
    def TPA(self,data,params,method='closest-K'):
        
        print("Converting a set of data points to simplicial complexes")
        if method=='persistent-homology':
            return self.rGraph(X=data,params=params)
        elif method=='closest-K':
            return self.kGraph(X=data,params=params)
        
    def rGraph(self,X,params):
         
        X= np.array(X)
        TRAIN = params['TRAIN']
        if TRAIN ==True:
            rip = ripser(X)
            zero_dimensional_homology = rip['dgms'][0][:,1][:-1]
            one_dimensional_homology = rip['dgms'][1][:,1][:-1]#optional
            mu = np.mean(zero_dimensional_homology)
            sigma = np.std(zero_dimensional_homology)
            self.r = mu+2.0*sigma
        
            self.data = X
            self.neigh = NearestNeighbors(radius=self.r, n_jobs=-1)
            self.neigh.fit(X)
            train_graph = self.neigh.radius_neighbors_graph(X,self.r)
            coo = train_graph.tocoo()
            return np.mat([coo.row, coo.col]).transpose()
        else:
            coo = self.neigh.radius_neighbors_graph(X,self.r).tocoo()
            return np.mat([coo.row+len(self.data), coo.col]).transpose()
            
    def kGraph(self,X,params):
        
        X= np.array(X)
        k = params['K']
        TRAIN = params['TRAIN']
        if TRAIN == True:
            self.data = X
            self.neigh = NearestNeighbors(n_neighbors=(k+1),n_jobs=-1)
            self.neigh.fit(X)
            train_graph = self.neigh.kneighbors_graph(X)
            coo = train_graph.tocoo()
            return np.mat([coo.row, coo.col]).transpose()
        else:
            coo = self.neigh.kneighbors_graph(X).tocoo()
            return np.mat([coo.row+len(self.data), coo.col ]).transpose()
        