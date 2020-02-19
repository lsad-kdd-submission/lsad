import pandas as pd
import numpy as np

def load(dataName):


    inputDim = 784
    normalNumber=80
    noise=5
    dataName='datasets/'+dataName
    train_P_df = pd.read_csv(dataName+'/'+'Train_P[%s].csv'%(normalNumber),sep=',')
    train_U_df = pd.read_csv(dataName+'/'+'Train_U[%s]_[%s].csv'%(normalNumber,noise),sep=',')
        
    test_df = pd.read_csv(dataName+'/'+'Test_set[%s].csv'%(normalNumber),sep=',')        
    trainU = train_U_df[[ '%s'%(x) for x in range(inputDim)] ].values   
    trainP = train_P_df[[ '%s'%(x) for x in range(inputDim)] ].values
        
    testX = test_df[[ '%s'%(x) for x in range(inputDim)] ].values
    testY= test_df['anomaly_label'].values

    N_list = train_U_df.loc[train_U_df['anomaly_label']==0].index
    A_list = train_U_df.loc[train_U_df['anomaly_label']==1].index
        
    if dataName =='multimnist':
        normalClass1 = int(normalNumber/10)
        normalClass2 = int(normalNumber%10)
            
        N_list1 = train_U_df.loc[train_U_df['type_label']==normalClass1].index
        N_list2 = train_U_df.loc[train_U_df['type_label']==normalClass2].index
    else:   
        N_list1 = N_list
        N_list2 = []

    return trainP,trainU,N_list,N_list1,N_list2,A_list,testX,testY
