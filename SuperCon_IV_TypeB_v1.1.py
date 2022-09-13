#-------------------------------------------------
# SuperCon_IV_TypeB.py ver1.1
#
# Copyright (c) 2021, Data PlatForm Center, NIMS
#
# This software is released under the MIT License.
#-------------------------------------------------
# -*- coding: utf-8 -*-


"""
機能：超電導のI-V特性のデータ構造化および可視化
機器：無冷媒低磁場物性測定装置 IV特性＠桜標準実験棟335

1) 半導体パラメーター（IV測定）から測定時間，電流，電圧のデータがtxtとして出力されるファイルを読み込む．
2) ヘッダー情報がない出力タイプ（Type Bと称する）から電界強度を計算し，I-V特性図を描画する．
　　　電圧間距離は0.5cmとセット
3) n値を自動的に計算する
4) Ic-B(T)図を作図する．

"""

__author__ = "Shigeyuki Matsunami"
__contact__ = "Matsunami.shigeyuki@nims.go.jp"
__license__ = "MIT"
__copyright__ = "National Institute for Materials Science, Japan"
__date__ = "2021/06/25"

# モジュール
import os
import glob
from natsort import natsorted
import re

# 数値処理用
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 可視化
from matplotlib import pyplot as plt

# util関数
# 実行処理#rawデータの読み込み
def read_files(extension):

    # 読み込みファイルの取得
    # "data"フォルダーに格納されたrawファイルを読み出すことにする
    path = 'data/*.' + extension
    input_files = natsorted(glob.glob(path))

    # 読み込みファイル数の取得
    number_of_files = len(input_files)

    # 拡張子を抜いたファイル名（出力用）
    output_name = [os.path.splitext(os.path.basename(p))[0] for p in input_files]
    
    print(output_name)

    return input_files, number_of_files, output_name

def get_magnetic_field(file):
 
    #ファイル名から磁場強度の読み出し
    pattern = "\d+T"
    
    field = re.search(pattern,file).group(0)
    magnetic_field = int(field.split('T')[0])

    return magnetic_field

def data_extract(file):

    #端子間距離は1cmと固定
    VV_length = 1

    ave = 0
    
    #数値部の取り出し
    df = pd.read_csv(file, sep='\t',names =['current','voltage'], header=1)
    
    #データ削除（熱起電力カット）
    df = df[:-5]
    df = df.query('current > 0').reset_index(inplace=False, drop=True)
    df = df.query('index > 20')
    
    #ベースライン・オフセット補正
    ave = df.query('20< index < 30').mean()   
    df['voltage'] = df['voltage']-ave['voltage']
    
    #電界強度の追加    
    df['Electric_field_strength'] = df['voltage']/float(VV_length)

    return df

def make_IV(df,file):

    #　図の設定 
    hfont = {'fontname': 'Arial'}
    fig, ax = plt.subplots(1,1, figsize=(7,7))
             
    X = df['current']
    Y = df['Electric_field_strength'] 
                
    ax.plot(X,Y,c='blue')

    #　作図のデザイン
    ax.set_xlabel('Current [A]',**hfont, fontsize = 18)
    ax.set_ylabel('Voltage [uV/cm]',**hfont, fontsize = 18)
    #ax.set_ylim(0,20)

    ax.tick_params(direction = "inout", length = 5, labelsize=14)
    ax.set_title(file,**hfont, fontsize = 16)
    ax.grid(which = "major", axis = "both", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)

    # y軸に目盛線を設定
    #ax.grid(which = "major", axis = "y", color = "blue", alpha = 0.8,linestyle = "--", linewidth = 0.1)
    
    #出力
    plt.savefig(file + '_I-V.png', dpi=300)
    plt.close()

def make_n_value(df,file):

    #　図の設定
    hfont = {'fontname': 'Arial'}
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    
    # 電圧領域として1～10V/cm範囲での電流電圧特性からn値を算出
   
    index_min = getNearestValue(df['voltage'], 1)
    index_max = getNearestValue(df['voltage'], 10)

    Ic = df['current'][index_min:index_max]
    Vc = df['voltage'][index_min:index_max]
    
    x = np.log(Ic,dtype = float).values.reshape(-1, 1)
    y = np.log(Vc,dtype = float)
    
    n = get_N_value(df)
    
    ax.plot(x,y,c='blue',marker="o",linestyle='--')

    #　作図のデザイン
    ax.set_xlabel('log (current) [A]',**hfont, fontsize = 18)
    ax.set_ylabel('log (voltage) [uV]',**hfont, fontsize = 18)
    ax.tick_params(direction = "inout", which = "both", length = 5, labelsize=14)
    ax.text(min(x),max(y)-0.1,'n value: {} '.format(n), **hfont, fontsize = 16)
    ax.grid(which = "major", axis = "x", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)
    ax.grid(which = "major", axis = "y", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)
    ax.set_title(file,**hfont, fontsize = 16)
    #ax.set_xlim(4,12)
    #ax.set_ylim(0.001,100)
    
    #出力
    plt.savefig(file + '_n-value.png', dpi=300)
    plt.close()

def make_Ic_B(df):

    #図の設定
    hfont = {'fontname': 'Arial'}
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    
            
    X = df['Magnetic_Field']
    Y = df['Ic'] 
         
    ax.plot(X,Y,c='blue',marker="o",linestyle='--')

    #作図のデザイン
    ax.set_yscale('log')
    ax.set_xlabel('B [T]',**hfont, fontsize = 18)
    ax.set_ylabel('Critical Current [A]',**hfont, fontsize = 18)
    ax.tick_params(direction = "inout", which = "both", length = 5, labelsize=14)
    ax.grid(which = "major", axis = "x", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)
    ax.grid(which = "minor", axis = "y", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)

    ax.set_xlim(0,20)
    ax.set_xticks( np.arange(0, 20.2, 2))
    ax.set_ylim(0.01,1000)
    ax.set_title('Ic - B(T)',**hfont, fontsize = 16)
    
    #出力
    plt.savefig('Ic-B.png', dpi=300)
    plt.close()
    
def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    
    return idx

def get_N_value(df):
    """
    概要: I-V特性からn値を算出する
    min: 電圧0.1Vのインデックスを取得
    max：電圧10Vのインデックスを取得
    return：log-log値の傾きをLinearRegressionで算出．その傾きをreturnする
    """
    
    # 電圧領域として1～10V/cm範囲での電流電圧特性からn値を算出
    index_min = getNearestValue(df['voltage'], 1)
    index_max = getNearestValue(df['voltage'], 10)

    Ic = df['current'][index_min:index_max]
    Vc = df['voltage'][index_min:index_max]
    
    x = np.log(Ic,dtype = float).values.reshape(-1, 1)
    y = np.log(Vc,dtype = float)
    
    solv = LinearRegression()
    solv.fit(x,y)
    
    return format(*solv.coef_, '.2f') 

# 実行処理
def main():
        
    # データの取得
    #　rawファイルは拡張子がない
    default_extension = 'dat'

    # ファイルの読み込み
    [files, f_num, fname] = read_files(default_extension)
    
    #　磁場，臨界電流，n値の初期設定
    Magnetic_Field = []
    Ic = []
    n_value = []

    #　I-V特性のマルチプロットの初期設定
    hfont = {'fontname': 'Arial'}
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    cmap = plt.get_cmap("tab10")
    
    
    # データ抽出（分割された複数ファイル）
    for i in range(f_num):
        
        try:
            #　数値データ部，磁場，n値の取得
            data = data_extract(files[i])
            
            #　数値データのcsv出力
            data.to_csv(fname[i]+'_extract.csv',index=False)
            
            #　可視化
            make_IV(data, fname[i])
            
            #磁場強度の取得
            mf = get_magnetic_field(files[i])
            Magnetic_Field.append(mf)
            
            #　臨界電流値の取得
            index = getNearestValue(data['Electric_field_strength'], 1)
            Ic.append(data['current'][index])
           
            #n値の取得
            n = get_N_value(data)
            n_value.append(n) 
            make_n_value(data, fname[i])
        
        except:
            continue

        
        #マルチプロット
        X = data['current']
        Y = data['Electric_field_strength'] 
        
        ax.plot(X,Y,color=cmap(i),label = fname[i])
    
    ax.set_xlabel('Current [A]',**hfont, fontsize = 18)
    ax.set_ylabel('Voltage [uV/cm]',**hfont, fontsize = 18)
    ax.set_ylim(0,40)
    ax.grid(which = "major", axis = "both", color = "black", alpha = 0.8,linestyle = "--", linewidth = 0.3)

    ax.tick_params(direction = "inout", length = 5, labelsize=14)
    ax.set_title('I-V-all',**hfont, fontsize = 16)
    ax.legend()
    fig.savefig('I-V-all.png', dpi=300)
    plt.close()
    
    # Ic-Bの作成と出力
    df_MF = pd.DataFrame(Magnetic_Field, columns = ['Magnetic_Field'])
    df_Ic = pd.DataFrame(Ic,columns = ['Ic'])
    df_n = pd.DataFrame(n_value,columns = ['n_value'])
    df = pd.concat([df_MF,df_Ic,df_n],axis=1)
    
    make_Ic_B(df)
    
    df.to_csv('Ic-MF-n.csv',index = False)

        
if __name__ == '__main__':
    main()