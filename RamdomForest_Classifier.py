# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:55:55 2018

@author: fukuda
"""

# 必要なライブラリのインポート
import time
import datetime
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import sklearn
import seaborn as sn
from sklearn.metrics import confusion_matrix
# 必要なライブラリのインポート
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV


def print_cmx(y_true, y_pred):
    #labels = sorted(list(set(y_true)))
    labels=['One','Two', 'Three', 'Four','Five']
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sn.heatmap(data=df_cmx, annot=True,square=True,cmap="Greys",fmt="d")
    #plt.savefig("confusion matrix.png", dpi=1200)
    plt.show()
    

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == '__main__':
    progress_s_time = datetime.datetime.today()
    print('実行開始時間：' + str( progress_s_time.strftime("%Y/%m/%d %H:%M:%S") ))
    progress_s_time = time.time()
    
    df1=pd.read_csv("output/{0}/OutputDataset_Dropna.csv".format(os.listdir( "./output" )[0]),index_col='Time',parse_dates=True)
    df2=pd.read_csv("output/{0}/OutputDataset_Dropna.csv".format(os.listdir( "./output" )[1]),index_col='Time',parse_dates=True)
    df=pd.concat([df1,df2])
    # 使用する列の選択
    
    #print(list(df.keys()))
    
    PCdatalist=[
            'AppName_Count', 
               # 'Curvature_075',
                'KeyTypeBack_Count',
              #'KeyTypeDel_Count',
              'KeyTypeEnter_Count',
              'KeyType_Count',
              'Lclick_Mean',
              'Lclick_Std',
              #'Mclick_Mean',
              'Mclick_Std',
              'MouseSpeed_Max',
              #'MouseWheel_Mean',
              'Mousedisplacement_Sum',
             # 'Rclick_Mean',
              #'Rclick_Std'
              ]
    
    
    Chairlist=['Rotation_Max', 'Rotation_Std', 'Rotation_Mean', 'Sag_mean']
    
    
    # 説明変数、目的変数
    X = df[PCdatalist+Chairlist ]
    tmp=list()
    Qnum="Q4"
    tmp=copy.deepcopy(df[Qnum])
    tmp[df[Qnum]==1]="One"
    tmp[df[Qnum]==2]="Two"
    tmp[df[Qnum]==3]="Three"
    tmp[df[Qnum]==4]="Four"
    tmp[df[Qnum]==5]="Five"
    y=tmp
    
    
    # 学習用、検証用データに分割
    from sklearn.cross_validation import train_test_split
    df_result=pd.DataFrame()
    for iii in range(1):
        
        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.3, random_state = 0)
        # モデル構築、パラメータはデフォルト
        
        print("################## ランダムフォレスト ##################")
        #ランダムフォレスト
        forest = RandomForestClassifier()
        forest.fit(X_train, y_train)
        # 予測値を計算
        y_test_pred = forest.predict(X_test)
        Accuracy_randomforest=sklearn.metrics.accuracy_score(y_test, y_test_pred)
        print('Accuracy: ' +str(Accuracy_randomforest))
        
        print("################## Confusion matrix ##################")
        print_cmx(y.values,forest.predict(X))
        # 出力
        
        print("################## 変数の重要度 ##################")
        print('Feature Importances:')
        feat_importance=pd.Series(forest.feature_importances_,index=X.keys())
        print(feat_importance.sort_values(ascending=False))
        
        #################################################
        if False:
            
            parameters = {
                'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
                'max_features'      : [3, 5, 10, 15, 20],
                'random_state'      : [0],
                'n_jobs'            : [1],
                'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
                'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
                }
            clf = GridSearchCV(RandomForestClassifier(), parameters)
            clf.fit(X_train, y_train)
            # 予測値を計算
            y_test_pred = clf.predict(X_test)
            
            # 出力
            print('Accuracy: ' +str(sklearn.metrics.accuracy_score(y_test, y_test_pred)))
            
            
        #################################################
        
        """
        
        print("################## SVM ##################")
        #ランダムフォレスト
        clf = svm.SVC(random_state=None)
        clf.fit(X_train, y_train)
        # 予測値を計算
        y_test_pred = clf.predict(X_test)
        
        # 出力
        Accuracy_svm=sklearn.metrics.accuracy_score(y_test, y_test_pred)
        print('Accuracy: ' +str(Accuracy_svm))
        
        print("################## アダブースト ##################")
        #アダブースト
        forest = AdaBoostClassifier()
        forest.fit(X_train, y_train)
        # 予測値を計算
        y_test_pred = forest.predict(X_test)
        
        # 出力
        Accuracy_AdaBoost=sklearn.metrics.accuracy_score(y_test, y_test_pred)
        print('Accuracy: ' +str(Accuracy_AdaBoost))
        
        print("################## GBDT ##################")
        GBDT = GradientBoostingClassifier()
        GBDT.fit(X_train, y_train)
        # 予測値を計算
        y_test_pred = forest.predict(X_test)
        
        # 出力
        Accuracy_GBDT=sklearn.metrics.accuracy_score(y_test, y_test_pred)
        print('Accuracy: ' +str(Accuracy_GBDT))    
        print(y.value_counts())
    

        #結果を保存
        tmpresult=pd.Series({"Accuracy_randomforest":Accuracy_randomforest,
                                "Accuracy_svm":Accuracy_svm,
                                "Accuracy_AdaBoost":Accuracy_AdaBoost,
                                "Accuracy_GBDT":Accuracy_GBDT,
                                
                                })
    
        df_result=df_result.append(tmpresult,ignore_index=True)
        """
    df_result.to_csv("5Grade Classification Accuracy.csv")
    print("########## 結果 ###########")
    print(df_result.mean())
    print(df_result.std())
    
    
    
    
    progress_e_time = time.time()
    progress_i_time = progress_e_time - progress_s_time
    print( '実行時間：' + str(round(progress_i_time,1)) + "秒" )
    
    
    
    
