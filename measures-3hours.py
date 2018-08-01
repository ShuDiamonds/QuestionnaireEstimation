# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:51:59 2017

@author: f1701
"""

import heartbeat as hb
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
import csv
sys.path.append('src')

import re

import time
import datetime

#from StringIO import StringIO

from scipy import signal
from scipy import fftpack, hamming


data_folder_path = "./Data-3hours/EmpaticaData/raw_data"
output_path = "./Data-3hours/EmpaticaData/ts_data"
data_folder_path2 = output_path

# 拡張子がcsvかチェックする
def check_file_extension( file_name ):
    ext = os.path.splitext( file_name )[1]
    if( ext == os.extsep + "csv" ):
        return True
    else:
        return False

def check_folder_name( folder_name ):
    if( folder_name == '.DS_Store' ):
        return False
    else:
        return True

def check_hiddenfolder_name( folder_name ):
    if( folder_name[0] == '.' ):
        return False
    else:
        return True
# フォルダが存在しない場合作成する
def not_exist_mkdir( output_path ):
    if( not os.path.exists(output_path) ):
        os.mkdir( output_path )

def trans_unixtime( unixtime_data ):
    #unixtimeをdatetimeに変換
    time_data = datetime.datetime.fromtimestamp( unixtime_data )
    return time_data

def trans_2unixtime( datetime_data ):
    #unixtimeをdatetimeに変換
    time_data = int(time.mktime(datetime_data.timetuple()))
    return time_data


# タイムスタンプの振り直し
# タイムスタンプの振り直し（開始）---------------------------------------------------------------
def timestampE4():
    not_exist_mkdir( output_path )

    files = os.listdir('.')
    folders1 = os.listdir( data_folder_path ) #ここにセッションごとのファイルネームを格納する
    #print("folders1:"+str(folders1)) #隠しファイルがあったのを確認した
    for folder_name1 in folders1:
        if check_hiddenfolder_name(folder_name1):
            not_exist_mkdir( output_path + os.sep + folder_name1 )
            if check_folder_name( folder_name1 ):
                print("degbug:"+ data_folder_path + os.sep + folder_name1)
                files = os.listdir( data_folder_path + os.sep + folder_name1 )
                usefiles = [ file_name for file_name in files if check_file_extension( file_name ) ]
                for file_name in usefiles:
                    sensor = os.path.splitext( file_name )[0]
                    if sensor == 'ACC' or sensor == 'BVP' or sensor == 'EDA' or sensor == 'HR' or sensor == 'IBI' or sensor == 'TEMP' :
                        mk_timestamp( folder_name1, file_name, sensor )



def mk_timestamp( folder_name1, file_name, sensor_type ):
    input_file_path = data_folder_path + os.sep + folder_name1 + os.sep + file_name
    output_file_path = output_path + os.sep + folder_name1 + os.sep + sensor_type + '_ts.csv'

    # ヘッダの書き込み
    fw = open( output_file_path, 'w' )
    writer = csv.writer(fw, lineterminator='\n')
    if sensor_type == 'ACC':
        writer.writerow( ['Time', 'ACC_X', 'ACC_Y', 'ACC_Z'] )
    else:
        writer.writerow( ['Time', sensor_type] )
    fw.close()

    file_rowlength = sum(1 for line in open( input_file_path ))
    #pbar = tqdm( total = file_rowlength )

    print('############# data #############')
    print(sensor_type)
    print(input_file_path)
    print(file_rowlength)
    print('################################')

    # データの書き込み
    fr = open( input_file_path, 'r' )
    reader = csv.reader( fr )


    fw = open( output_file_path, 'a' )
    writer = csv.writer(fw, lineterminator='\n')

    for raw_num, raw_data in enumerate( reader ):
        if sensor_type == 'IBI':
            if raw_num == 0:
                start_time = float( raw_data[0] )
            else:
                delta_time = float( raw_data[0] )
                ibi_value = raw_data[1]

                timestamp = start_time + delta_time
                writer.writerow( map(float, [timestamp, ibi_value]) )

        elif sensor_type == 'ACC':
            if raw_num == 0:
                start_time = float( raw_data[0] )
            elif raw_num == 1:
                #sampling_rate = raw_data[0]
                delta_time = 1.0 / float( raw_data[0] )
            else:

                timestamp = start_time + (raw_num - 2) * delta_time
                writer.writerow( map(float, [timestamp, raw_data[0], raw_data[1], raw_data[2]]) )
        else:
            if raw_num == 0:
                start_time = float( raw_data[0] )
            elif raw_num == 1:
                delta_time = 1.0 / float( raw_data[0] )
            else:

                timestamp = start_time + (raw_num - 2) * delta_time
                writer.writerow( map(float, [timestamp, raw_data[0]]) )

    fw.close()
    fr.close()

# タイムスタンプの振り直し（終了）---------------------------------------------------------------


def psd_LHratio( dataframe ,sanpleraito=64):
    # LF : 0.04 - 0.15 Hz
    # HF : 0.15 - 0.40 Hz
    try:
        
        ################# Add ###############
        sig = dataframe
        # パワースペクトル密度を計算
        Nsample = sig.size
        time_step = 1.0 / sanpleraito  # sec：サンプリング間隔
        time_vec = np.arange(Nsample) * time_step
        sample_freq = fftpack.fftfreq( sig.size, d=time_step )  # Hz：フーリエ変換の横軸
        sig_fft = fftpack.fft( sig )  # フーリエ変換
        pidxs = np.where( sample_freq > 0 )  # プラスだけ抽出
        freqs, power = sample_freq[pidxs], np.abs(sig_fft)[pidxs]
        freq = freqs[power.argmax()]  # Hz：ピーク周波数抽出
        
        psd_df = pd.DataFrame({'freqs':freqs, 'power':power})
            
        LF_sum = 0
        HF_sum = 0
        for num in range( psd_df.shape[0] ):
            if 0.04 <= psd_df.ix[num,'freqs'] and psd_df.ix[num,'freqs'] <= 0.15:
                LF_sum = LF_sum + abs(psd_df.ix[num,'power'])
            elif 0.15 <= psd_df.ix[num,'freqs'] and psd_df.ix[num,'freqs'] <= 0.4:
                HF_sum = HF_sum + abs(psd_df.ix[num,'power'])
            elif 0.5 <= psd_df.ix[num,'freqs']:
                break
        
        LHratio = LF_sum / HF_sum
        
        return LHratio
    except Exception as e:
        print('=== エラー内容 ===')
        print('type:' + str(type(e)))
        print('args:' + str(e.args))
        print('e自身:' + str(e))
        return np.NaN

if __name__ == '__main__':
    progress_s_time = datetime.datetime.today()
    print('実行開始時間：' + str( progress_s_time.strftime("%Y/%m/%d %H:%M:%S") ))
    progress_s_time = time.time()
    
    #Empaticaのデータのタイムスタンプの付加
    #timestampE4()
    
    
    ############データの読み込み############
    
    #####BVPデータの読み込み
    EmpaticaDatafilepath=output_path+os.sep+os.listdir( data_folder_path )[0]+"/BVP_ts.csv"
    df = pd.read_csv(EmpaticaDatafilepath)
    DateTimeArray=[trans_unixtime(Time) for Time in  df.loc[:,"Time"]] #unixtimeの変換
    BVP_df = pd.DataFrame({'Time':DateTimeArray, 'BVP':df.loc[:,"BVP"] })
    BVP_df=BVP_df.set_index('Time')
    #####IBIデータの読み込み
    EmpaticaDatafilepath_ibi=output_path+os.sep+os.listdir( data_folder_path )[0]+"/IBI_ts.csv"
    df = pd.read_csv(EmpaticaDatafilepath_ibi)
    DateTimeArray=[trans_unixtime(Time) for Time in  df.loc[:,"Time"]] #unixtimeの変換
    IBI_df = pd.DataFrame({'Time':DateTimeArray, 'IBI':df.loc[:,"IBI"] ,"UnixTime":df.loc[:,"Time"] })
    IBI_df=IBI_df.set_index('Time')
    #####HRデータの読み込み
    EmpaticaDatafilepath_hr=output_path+os.sep+os.listdir( data_folder_path )[0]+"/HR_ts.csv"
    df = pd.read_csv(EmpaticaDatafilepath_hr)
    DateTimeArray=[trans_unixtime(Time) for Time in  df.loc[:,"Time"]] #unixtimeの変換
    HR_df = pd.DataFrame({'Time':DateTimeArray, 'HR':df.loc[:,"HR"] ,"UnixTime":df.loc[:,"Time"] })
    HR_df=HR_df.set_index('Time')
    
    #####DeskDataデータの読み込み
    DeskData_parser = lambda d: pd.datetime.strptime(d, "%Y/%m/%d %H:%M:%S")
    DeskDatafilepath="./Data-3hours/DeskData"+os.sep+os.listdir( "./Data-3hours/DeskData" )[0]+os.sep+os.listdir( "./Data-3hours/DeskData/"+os.listdir( "./Data-3hours/DeskData" )[0] )[0]
    DeskData_df=pd.read_csv(DeskDatafilepath,index_col='TimeStamp',parse_dates=True, date_parser=DeskData_parser)
    
    #####ChairDataデータの読み込み
    ChairData_parser = lambda d: pd.datetime.strptime(d, "%Y/%m/%d %H:%M:%S:%f")
    ChairDatafilepath="./Data-3hours/ChairData/"+os.listdir( "./Data-3hours/ChairData/" )[0]
    tmp1=pd.read_csv(ChairDatafilepath+"/logging.csv",index_col='Timestamp',parse_dates=True, date_parser=ChairData_parser)    #loggingファイルの読み込み
    tmp2=pd.read_csv(ChairDatafilepath+"/acceleration.csv",index_col='TimeStamp',parse_dates=True, date_parser=ChairData_parser)    #accelerationファイルの読み込み
    tmp3=pd.read_csv(ChairDatafilepath+"/gyro.csv",index_col='TimeStamp',parse_dates=True, date_parser=ChairData_parser)    #gyroファイルの読み込み
    Chair_df=tmp1.join(tmp2)
    Chair_df=Chair_df.join(tmp3)
    
    
    not_exist_mkdir( "./image" )
    MesuresDataFrame=pd.DataFrame()
    OutputDataset=pd.DataFrame()
    
    #時間集計関係の変数の定義
    WindowTime=offsets.Minute(5) #時系列データの窓は5分
    SiftTime=30                  #３０秒ずつに窓をスライドさせる
    #それぞれのデータのうち一番記録開始が遅かったものの時間を基準にする
    DataEarlestTime=trans_unixtime(max(trans_2unixtime(BVP_df.index[0]),trans_2unixtime(DeskData_df.index[0]),trans_2unixtime(Chair_df.index[0])))
    #それぞれのデータのうち一番記録終了が早かった物の時間を基準にする
    DataENDTime=trans_unixtime(min(trans_2unixtime(BVP_df.index[-1]),trans_2unixtime(DeskData_df.index[-1]),trans_2unixtime(Chair_df.index[-1])))
    
    print("program will pcocess " +str( (DataENDTime-DataEarlestTime).seconds ) +"sec data")
    i=0
    
    
    for cnt_t in range(0,(DataENDTime-DataEarlestTime).seconds,SiftTime):     #単位は秒
        
        #try:
            ## 進捗を表示
            i=i+1
            print("#################################################")
            print("progress: "+str( round(i*100.0/len(range(0,(DataENDTime-DataEarlestTime).seconds,30)),1) )+"% done...")
            print("i="+str(i)+", len = "+str(len(range(0,(DataENDTime-DataEarlestTime).seconds,30))))
            
            Dt_start = DataEarlestTime+offsets.Second(cnt_t)
            Dt_End = Dt_start + WindowTime
            CHARstartTime=Dt_start.strftime('%Y-%m-%d  %H:%M:%S')
            CHARendTime=Dt_End.strftime('%Y-%m-%d  %H:%M:%S')
            
            if Dt_End>DataENDTime:
                print("############データは１時間分処理されませんでした###############")
                break
            ############### LF/HF ##############
            hb.working_data.clear()
            hb.measures.clear()
            fs=63.9999999999    #64.0だと動かないので仕方なし
            measures1 = hb.process(BVP_df[CHARstartTime:CHARendTime]['BVP'].values, fs)
            #plot_object = hb.plotter(show=False)
            #plot_object.savefig(('./image/'+'plot_'+Dt_start.strftime('%Y-%m-%d %H_%M_%S')+'.jpg'),dpi=500) #saves the plot as JPEG image.
            #plot_object.close()
            del measures1["nn20"]
            del measures1["nn50"]
            measures1["Time"]=Dt_End
            MesuresDataFrame=MesuresDataFrame.append(pd.Series(measures1.values(),index=measures1.keys() ),ignore_index=True)
            print(CHARendTime+" bpm: "+ str(measures1['bpm']))
            print(CHARendTime+" lf/hf: "+ str(measures1['lf/hf']))
            
            
            ############### IBIからのLF/HF　の算出 ###########
            tmpibi=IBI_df[CHARstartTime:CHARendTime]['IBI']
            tmpibi=tmpibi.resample('1000L',how='mean').fillna(method='ffill')
            
            #x_ibi=IBI_df[CHARstartTime:CHARendTime]['UnixTime']
            #functionibi = interp1d(x_ibi, tmpibi, kind='cubic')    # ３次スプライン補間
            #x_ibi_new=np.linspace(x_ibi[0],x_ibi[-1],x_ibi[-1])
            #print("ibi data num: "+str(tmpibi.size))
            
            ############### デスクデータの切り出し ##############
            tmpData=DeskData_df[CHARstartTime:CHARendTime]
            tmpData1=Chair_df[CHARstartTime:CHARendTime]
            rotation=np.abs(tmpData1["5_Gz_Ave"])
            
            
            ############## 特徴量の算出 ####################
            tmpdeskdata=pd.Series({
                    "Time":Dt_End,
                    "LHratio_Amenomori":psd_LHratio(BVP_df[CHARstartTime:CHARendTime]['BVP']),
                    "LHratio_IBI":psd_LHratio(tmpibi,1.0),
                    "lf/hf":measures1['lf/hf'],
                    "SDNN":measures1['sdnn'],
                    "HR_Mean":HR_df[CHARstartTime:CHARendTime]['HR'].mean(),
                    "HR_Std":HR_df[CHARstartTime:CHARendTime]['HR'].std(),
                    "Lclick_Mean":tmpData["clickCNTL"].mean(),
                    "Lclick_Std":tmpData["clickCNTL"].std(),
                    "Rclick_Mean":tmpData["clickCNTR"].mean(),
                    "Rclick_Std":tmpData["clickCNTR"].std(),
                    "Mclick_Mean":tmpData["clickCNTM"].mean(),
                    "Mclick_Std":tmpData["clickCNTM"].std(),
                    "Mousedisplacement_Sum":tmpData["MouseDisplacement"].sum(),
                    "MouseSpeed_Max":tmpData["MouseSpeedMax"].max(),
                    "MouseWheel_Mean":tmpData["MouseWheelAmount"].mean(),
                    "KeyType_Count":tmpData["KeyTypeCNT"].sum(),
                    "KeyTypeDel_Count":tmpData["KeyTypeDelCNT"].sum(),
                    "KeyTypeBack_Count":tmpData["KeyTypeBackCNT"].sum(),
                    "KeyTypeEnter_Count":tmpData["KeyTypeEnterCNT"].sum(),
                    "Curvature_075":DeskData_df["Curvature"][DeskData_df["Curvature"]>DeskData_df["Curvature"].quantile(.75)].count(),
                    "AppName_Count":DeskData_df["AppName"].value_counts().count(),
                    "WeightEffort_Mean":tmpData["WeightEffort"].mean(),
                    "WeightEffort_Std":tmpData["WeightEffort"].std(),
                    "TimeEffort_Mean":tmpData["TimeEffort"].mean(),
                    "TimeEffort_Std":tmpData["TimeEffort"].std(),
                    "SpaceEffort_Mean":tmpData["SpaceEffort"].mean(),
                    "SpaceEffort_Std":tmpData["SpaceEffort"].std(),
                    "TableShape_Mean":tmpData["DoorShape"].mean(),
                    "TableShape_Std":tmpData["DoorShape"].std(),
                    "DoorShape_Mean":tmpData["DoorShape"].mean(),
                    "DoorShape_Std":tmpData["DoorShape"].std(),
                    "WheelShape_Mean":tmpData["WheelShape"].mean(),
                    "WheelShape_Std":tmpData["WheelShape"].std(),
                    "JawOpen_Mean":tmpData["LM_JawOpen"].mean(),
                    "JawOpen_Std":tmpData["LM_JawOpen"].std(),
                    "LeftcheekPuff_Mean":tmpData["LM_LeftcheekPuff"].mean(),
                    "LeftcheekPuff_Std":tmpData["LM_LeftcheekPuff"].std(),
                    "LipCornerDepressorRight_Mean":tmpData["LM_LipCornerDepressorRight"].mean(),
                    "LipCornerDepressorRight_Std":tmpData["LM_LipCornerDepressorRight"].std(),
                    "LipCornerDepressorLeft_Mean":tmpData["LM_LipCornerDepressorLeft"].mean(),
                    "LipCornerDepressorLeft_Std":tmpData["LM_LipCornerDepressorLeft"].std(),
                    "LM_LipCornerPullerRight_Mean":tmpData["LM_LipCornerPullerRight"].mean(),
                    "LM_LipCornerPullerRight_Std":tmpData["LM_LipCornerPullerRight"].std(),
                    "LipCornerPullerLeft_Mean":tmpData["LM_LipCornerPullerLeft"].mean(),
                    "LipCornerPullerLeft_Std":tmpData["LM_LipCornerPullerLeft"].std(),
                    "RightcheekPuff_Mean":tmpData["LM_RightcheekPuff"].mean(),
                    "RightcheekPuff_Std":tmpData["LM_RightcheekPuff"].std(),
                    "LeftcheekPufff_Mean":tmpData["LM_LeftcheekPufff"].mean(),
                    "LeftcheekPufff_Std":tmpData["LM_LeftcheekPufff"].std(),
                    "RighteyebrowLowerer_Mean":tmpData["LM_RighteyebrowLowerer"].mean(),
                    "RighteyebrowLowerer_Std":tmpData["LM_RighteyebrowLowerer"].std(),
                    "LefteyebrowLowerer_Mean":tmpData["LM_LefteyebrowLowerer"].mean(),
                    "LefteyebrowLowerer_Std":tmpData["LM_LefteyebrowLowerer"].std(),
                    
                    "Sag_mean":tmpData1["7_Ay_Ave"].mean(),
                    "Sag_075":tmpData1["7_Ay_Ave"][tmpData1["7_Ay_Ave"]>tmpData1["7_Ay_Ave"].quantile(.75)].count(),
                    "Rotation_Sum":rotation.sum(),
                    "Rotation_Max":rotation.max(),
                    "Rotation_Std":rotation.std(),
                    "HowSit_count":tmpData1["Posture"].value_counts().count()
                    })
            
            OutputDataset=OutputDataset.append(tmpdeskdata,ignore_index=True)
            """except Exception as e:
            print('=== エラー内容 ===')
            print('type:' + str(type(e)))
            print('args:' + str(e.args))
            print('e自身:' + str(e))
            """
    ##### 正規化関数の定義
    zscore = lambda x: (x - x.mean()) / x.std()
    OutputDataset=OutputDataset.set_index("Time")
    OutputDataset.to_csv("OutputDataset-3hours.csv")
    OutputDataset1=OutputDataset.dropna()
    OutputDataset1.to_csv("OutputDataset_Dropna-3hours.csv")
    OutputDataset2=OutputDataset.fillna(OutputDataset.mean())
    #OutputDataset2=OutputDataset2.apply(lambda x: (x-x.mean())/x.std(),axis=0).fillna(0)
    OutputDataset2.to_csv("OutputDataset_No-missing-3hours.csv")
    #LF/HFデータの正規化
    
    OutputDataset["LHratio_IBI"]=zscore(OutputDataset["LHratio_IBI"])
    OutputDataset["LHratio_Amenomori"]=zscore(OutputDataset["LHratio_Amenomori"])
    OutputDataset["lf/hf"]=zscore(OutputDataset["lf/hf"])
    
    MesuresDataFrame=MesuresDataFrame.set_index("Time")
    MesuresDataFrame.to_csv("MesuresDataFrame-3hours.csv")
    progress_e_time = time.time()
    progress_i_time = progress_e_time - progress_s_time
    print("**********************")
    print("##############　欠損値の割合　############")
    print(OutputDataset.isnull().sum()*100/len(OutputDataset))
    print("######################################")
    
    
    print("Data length: "+str(len(OutputDataset)))
    print( '実行時間：' + str(round(progress_i_time,1)) + "秒" )
    print("Data Start Time"+str(DataEarlestTime))
    print("Data End Time"+str(DataENDTime))
    print("Data Duration: "+str(DataENDTime-DataEarlestTime))
    #欠損値の割合のCSV出力
    (OutputDataset.isnull().sum()*100/len(OutputDataset)).to_csv("About Missing Values.csv-3hours")
    #実行結果をCSV出力
    pd.Series({"Data Start Time":DataEarlestTime,
               "Data End Time":DataENDTime,
               "Data Duration":(DataENDTime-DataEarlestTime), 
                  "Data length":len(OutputDataset),
                  '実行時間：':(str(progress_i_time)),
                  
                  }).to_csv("Report.csv")
    
    
    # 画像の出力と保存
    OutputDataset["LHratio_IBI"].plot()
    OutputDataset["SDNN"].plot()
    OutputDataset["LHratio_Amenomori"].plot()
    ax =OutputDataset["lf/hf"].plot()
    fig = ax.get_figure()
    fig.savefig("LFHF_mult.png-3hours", dpi=600)
    
