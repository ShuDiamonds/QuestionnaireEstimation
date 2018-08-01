# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:03:33 2018

@author: fukuda
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:57:34 2018

@author: fukuda
"""
import pandas as pd
import pandas.tseries.offsets as offsets
import copy
from scipy.interpolate import interp1d

import pdb
import scipy
from scipy import interpolate
import os
#import ConfigParser
import math
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('src')

import re

import time
import datetime

def trans_unixtime( unixtime_data ):
    #unixtimeをdatetimeに変換
    time_data = datetime.datetime.fromtimestamp( unixtime_data )
    return time_data

def trans_2unixtime( datetime_data ):
    #unixtimeをdatetimeに変換
    time_data = int(time.mktime(datetime_data.timetuple()))
    return time_data
def not_exist_mkdir( output_path ):
    if( not os.path.exists(output_path) ):
        os.mkdir( output_path )

data_folder_path = "./data/"

if __name__ == '__main__':
    
    progress_s_time = datetime.datetime.today()
    print('実行開始時間：' + str( progress_s_time.strftime("%Y/%m/%d %H:%M:%S") ))
    progress_s_time = time.time()
    not_exist_mkdir("./output")
    
    for username in os.listdir( data_folder_path ):
        for datefilepath in os.listdir( data_folder_path+username ):
            output_path = "./output/"+username+"/"+datefilepath+"/"
            input_path = data_folder_path+username+"/"+datefilepath+"/"
            not_exist_mkdir("./output/"+username+"/")
            not_exist_mkdir(output_path)
            #デスクワークデータの読み込み
            DeskworkDatafilepath=input_path+"output.csv"
            DeskData_parser = lambda d: pd.datetime.strptime(d, "%Y/%m/%d %H:%M:%S")
            DeskworkData_df=pd.read_csv(DeskworkDatafilepath,index_col='TimeStamp',parse_dates=True, date_parser=DeskData_parser,keep_default_na=False,na_values="NaN")
            #デスクワークデータの読み込み
            QuestionnaireDatafilepath=input_path+"Questionnaire.csv"
            QuestionnaireData_df=pd.read_csv(QuestionnaireDatafilepath,index_col='TimeStamp',parse_dates=True, date_parser=DeskData_parser)
            
            OutputDataset=pd.DataFrame()
        
        
            for QestionaireTime in QuestionnaireData_df.index:
                StartTime=QestionaireTime+datetime.timedelta(minutes=-15)
                EndTime=QestionaireTime
                print(StartTime)
                tmpData=DeskworkData_df[StartTime:EndTime]
                tmpData=tmpData.drop(["AppName","Memo","Unnamed: 23"],axis=1).astype(float) #文字列を削除する（NaNを含む列がobject型として読み込まれるから、文字列を含む列を全部消してそのあとastypeでfloatに変換してる）
                tmpFeaturesData=pd.Series({
                            "Time":EndTime,
                            "Q1":QuestionnaireData_df[QestionaireTime.strftime('%Y-%m-%d %H:%M:%S')].values[0][0],
                            "Q2":QuestionnaireData_df[QestionaireTime.strftime('%Y-%m-%d %H:%M:%S')].values[0][1],
                            "Q3":QuestionnaireData_df[QestionaireTime.strftime('%Y-%m-%d %H:%M:%S')].values[0][2],
                            "Q4":QuestionnaireData_df[QestionaireTime.strftime('%Y-%m-%d %H:%M:%S')].values[0][3],
                            "Lclick_Mean":tmpData["clickCNTL"].mean(),
                            "Lclick_Std":tmpData["clickCNTL"].std(),
                            "Lclick_oneclick":tmpData["clickCNTL"][tmpData["clickCNTL"]==1].count(),
                            "Lclick_doubleclick":tmpData["clickCNTL"][tmpData["clickCNTL"]==2].count(),
                            "Rclick_Mean":tmpData["clickCNTR"].mean(),
                            "Rclick_Std":tmpData["clickCNTR"].std(),
                            "Mclick_Mean":tmpData["clickCNTM"].mean(),
                            "Mclick_Std":tmpData["clickCNTM"].std(),
                            "Mousedisplacement_Sum":tmpData["MouseDisplacement"].sum(),
                            "Mousedisplacement_lower50": tmpData["MouseDisplacement"][tmpData["MouseDisplacement"]<50].count()/len(tmpData["MouseDisplacement"]),    
                            "MouseSpeed_Max":tmpData["MouseSpeedMax"].max(),
                            "MouseWheel_Mean":tmpData["MouseWheelAmount"].mean(),
                            "KeyType_Count":tmpData["KeyTypeCNT"].sum(),
                            "KeyTypeDel_Count":tmpData["KeyTypeDelCNT"].sum(),
                            "KeyTypeBack_Count":tmpData["KeyTypeBackCNT"].sum(),
                            "KeyTypeEnter_Count":tmpData["KeyTypeEnterCNT"].sum(),
                            "mistyping_Count":(tmpData["KeyTypeBackCNT"]+tmpData["KeyTypeDelCNT"]).value_counts().count(),
                            
                            "Sag_mean":tmpData["Sag"].dropna().mean(),
                            "Sag_std":tmpData["Sag"].dropna().std(),
                            "Rotation_Mean":tmpData["Rotation"].dropna().mean(),
                            "Rotation_Max":tmpData["Rotation"].dropna().max(),
                            "Rotation_Std":tmpData["Rotation"].dropna().std(),
                            "Rotation_lower05": tmpData["Rotation"].dropna()[np.abs(tmpData["Rotation"].dropna())<0.5].count()/len(tmpData["Rotation"].dropna()),
                            
                            "Compass_mean":tmpData["Compass"].dropna().mean(),
                            "Compass_std":tmpData["Compass"].dropna().std(),
                            "Posture_RightLeft_Mean":tmpData["Posture_RightLeft"].dropna().mean(),
                            "Posture_RightLeft_Max":tmpData["Posture_RightLeft"].dropna().max(),
                            "Posture_RightLeft_Std":tmpData["Posture_RightLeft"].dropna().std(),
                            "Posture_Rear_Mean":tmpData["Posture_Rear"].dropna().mean(),
                            "Posture_Rear_Max":tmpData["Posture_Rear"].dropna().max(),
                            "Posture_Rear_Std":tmpData["Posture_Rear"].dropna().std(),
                            "Posture_Front_Mean":tmpData["Posture_Front"].dropna().mean(),
                            "Posture_Front_Max":tmpData["Posture_Front"].dropna().max(),
                            "Posture_Front_Std":tmpData["Posture_Front"].dropna().std()
                            })
                    
                OutputDataset=OutputDataset.append(tmpFeaturesData,ignore_index=True)
                
            ##### 正規化関数の定義
            zscore = lambda x: (x - x.mean()) / x.std()
            OutputDataset=OutputDataset.set_index("Time")
            OutputDataset.to_csv(output_path+"OutputDataset.csv")
            OutputDataset1=OutputDataset.dropna()
            OutputDataset1.to_csv(output_path+"OutputDataset_Dropna.csv")
            OutputDataset2=OutputDataset.fillna(OutputDataset.mean())
            #OutputDataset2=OutputDataset2.apply(lambda x: (x-x.mean())/x.std(),axis=0).fillna(0)
            OutputDataset2.to_csv(output_path+"OutputDataset_No-missing.csv")
            
            print("**********************")
            print("##############　欠損値の割合　############")
            print("############# "+username+" ############")
            print(OutputDataset.isnull().sum()*100/len(OutputDataset))
            #欠損値の割合のCSV出力
            (OutputDataset.isnull().sum()*100/len(OutputDataset)).to_csv(output_path+"About Missing Values.csv")
            
        print("######################################")
        progress_e_time = time.time()
        progress_i_time = progress_e_time - progress_s_time
        DataEarlestTime=QuestionnaireData_df.index[0]
        DataENDTime=QuestionnaireData_df.tail(1).index[0]
        print("Data length: "+str(len(OutputDataset)))
        print( '実行時間：' + str(round(progress_i_time,1)) + "秒" )
        print("Data Start Time"+str(DataEarlestTime))
        print("Data End Time"+str(DataENDTime))
        print("Data Duration: "+str(DataENDTime-DataEarlestTime))
        
        #実行結果をCSV出力
        pd.Series({"Data Start Time":DataEarlestTime,
                   "Data End Time":DataENDTime,
                   "Data Duration":(DataENDTime-DataEarlestTime), 
                      "Data length":len(OutputDataset),
                      '実行時間：':(str(progress_i_time)),
                      
                      }).to_csv("Report.csv")
    
    


    
    
    
    
    