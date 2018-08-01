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

labels=['One','Two', 'Three', 'Four','Five']
def print_cmx(y_true, y_pred):
    #labels = sorted(list(set(y_true)))
    
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
              "mistyping_Count",
              'Lclick_Mean',
              'Lclick_Std',
              "Lclick_oneclick",
              "Lclick_doubleclick",
              "Rclick_Mean",
              "Rclick_Std",
              #'Mclick_Mean',
              'Mclick_Std',
              'MouseSpeed_Max',
              'MouseWheel_Mean',
              'Mousedisplacement_Sum',
              "Mousedisplacement_lower50"
              ]
    
    Chairlist=['Rotation_Max', 'Rotation_Std', 'Rotation_Mean',"Rotation_lower05", 'Sag_mean',"Sag_std"]
    
    
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
    df_feat_importance=pd.DataFrame()
    df_cmx=pd.DataFrame(np.zeros((5,5),dtype=int),index=labels, columns=labels)
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 10, shuffle = True)
    for train_index, test_index in kf.split(X):
        #print(train_index, test_index)
        (X_train, X_test, y_train, y_test) = (X.iloc[train_index],X.iloc[test_index],y.iloc[train_index],y.iloc[test_index])
        
        # モデル構築、パラメータはデフォルト
        
        
        
        print("################## ランダムフォレスト ##################")
        #ランダムフォレスト
        forest = RandomForestClassifier(n_estimators=150, 
        criterion='gini', 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        max_features='auto', 
        max_leaf_nodes=None, 
        bootstrap=True, 
        oob_score=False, 
        n_jobs=1, 
        random_state=1, 
        verbose=0
        )
        forest.fit(X_train, y_train)
            
        # 予測値を計算
        y_test_pred = forest.predict(X_test)
        Accuracy_randomforest=sklearn.metrics.accuracy_score(y_test, y_test_pred)
        Precision_randomforest=sklearn.metrics.precision_score(y_test, y_test_pred,average="weighted")
        Recall_randomforest=sklearn.metrics.recall_score(y_test, y_test_pred,average="weighted")
        Fmeasure_randomforest=sklearn.metrics.f1_score(y_test, y_test_pred,average="weighted")
        # 出力
        print('Accuracy: ' +str(Accuracy_randomforest))
        print('Precision: ' +str(Precision_randomforest))
        print('Recall: ' +str(Recall_randomforest))
        print('Fmeasure: ' +str(Fmeasure_randomforest))
        
        print("################## Confusion matrix ##################")
        #print_cmx(y,forest.predict(X))
        # 出力
        
        #labels = sorted(list(set(y_true)))
        cmx_data = confusion_matrix(y_test, y_test_pred, labels=labels)
        
        df_cmx = df_cmx + pd.DataFrame(cmx_data, index=labels, columns=labels)
        
        
        print("################## 変数の重要度 ##################")
        print('Feature Importances:')
        feat_importance=pd.Series(forest.feature_importances_,index=X.keys())
        print(feat_importance.sort_values(ascending=False))
        df_feat_importance=df_feat_importance.append(feat_importance,ignore_index=True)
        
            
        #################################################
        #結果を保存
        tmpresult=pd.Series({"Accuracy":Accuracy_randomforest,
                             "Precision":Precision_randomforest,
                             "Recall":Recall_randomforest,
                             "Fmeasure":Fmeasure_randomforest,
                                })
    
        df_result=df_result.append(tmpresult,ignore_index=True)
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
    
    print("################## 変数の重要度 ##################")
    print('Feature Importances:')
    feat_importance=pd.Series(forest.feature_importances_,index=X.keys())
    print(df_feat_importance.mean().sort_values(ascending=False))
    
    print("################## Confusion matrix ##################")    
    plt.figure(figsize = (10,7))
    sn.heatmap(data=df_cmx, annot=True,square=True,cmap="Greys",fmt="d")
    #plt.savefig("confusion matrix.png", dpi=1200)
    plt.show()
    
    plt.figure(figsize = (10,7))
    df_cmx2=copy.copy(df_cmx)
    a=[]
    for key,row in df_cmx.iterrows():
        print((row/df_cmx.sum(axis=1)[key]).values)
        a.append((row/df_cmx.sum(axis=1)[key]).values)
    df_cmx2=pd.DataFrame(a,index=labels, columns=labels)
    #df_cmx2 = df_cmx / df_cmx.sum(axis=1)
    #df_cmx = df_cmx.applymap(np.int64)
    sn.heatmap(data=df_cmx2, annot=True,annot_kws={'size': 15},square=True,cmap="Greys",fmt=".0%",cbar=None)
    plt.tick_params(labelsize=20)
    plt.savefig("confusion matrix-3grades.png", dpi=300)
    plt.show()
    
    
    
    progress_e_time = time.time()
    progress_i_time = progress_e_time - progress_s_time
    print( '実行時間：' + str(round(progress_i_time,1)) + "秒" )
    
    
    
    
