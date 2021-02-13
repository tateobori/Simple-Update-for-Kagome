# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():


    filename = 'D6-kagome_iso.txt'
    data = np.loadtxt(filename,delimiter='\t').T
    Hz = data[0]
    Hz_norm = data[1]
    M = data[2]
    M_delta = data[3]
    M_ja = data[4]

    # 散布図
    plt.scatter(Hz_norm,M, color="r", label=r'$M_{tot}=M_{\Delta}+M_{J_A}$')
    plt.scatter(Hz_norm,M_delta, color="blue", label=r'$M_{\Delta}$')
    plt.scatter(Hz_norm,M_ja, color="black", label=r'$M_{J_A}$')
    plt.hlines([1.0/3.0], xmin=0.0, xmax=100.0, color="black", linestyles='dashed')
    plt.hlines([2.0/3.0], xmin=0.0, xmax=100.0, color="black", linestyles='dashed')
    #plt.axhline(y=1.0/3.0, xmin=0.0, xmax=100, color="black", linestyles='dashed')
    #plt.scatter(x2,y2,c='#00cc00')

    # 軸名とタイトル
    plt.xlabel(r'$H_z$(T)') #x軸名。Latex形式で書ける
    plt.ylabel(r'$M_z/M_{sat}$') #y軸名

    # 表示
    plt.grid()
    plt.legend()
    plt.savefig('figure.png')
    plt.show()
    plt.close()

def bkh12_plot():


    filename = 'D6-BKH.txt'
    data = np.loadtxt(filename,delimiter='\t').T
    Hz = data[0]
    M1 = data[1]
    M2 = data[2]
    M3 = data[3]
    M4 = data[4]

    # 散布図
    plt.scatter(Hz,M1, label=r'$J_{\bigtriangledown}/J_{\bigtriangleup}=0.6$', marker="o")
    plt.scatter(Hz,M2, label=r'$J_{\bigtriangledown}/J_{\bigtriangleup}=0.3$', marker="d")
    plt.scatter(Hz,M3, label=r'$J_{\bigtriangledown}/J_{\bigtriangleup}=0.1$', marker="*")
    plt.scatter(Hz,M4, label=r'$J_{\bigtriangledown}/J_{\bigtriangleup}=0.01$', marker="^")

    #plt.axhline(y=1.0/3.0, xmin=0.0, xmax=100, color="black", linestyles='dashed')
    #plt.scatter(x2,y2,c='#00cc00')

    # 軸名とタイトル
    plt.xlabel(r'$H_z$') #x軸名。Latex形式で書ける
    plt.ylabel(r'$M_z/M_{sat}$') #y軸名

    # 表示
    plt.grid()
    plt.legend()
    plt.savefig('BKH.png')
    plt.show()
    plt.close()

def bkh32_plot():


    filename = 'D6-BKH32.txt'
    data = np.loadtxt(filename,delimiter='\t').T
    Hz = data[0]
    M1 = data[1]
    M2 = data[2]
    M3 = data[3]
    M4 = data[4]

    # 散布図
    plt.scatter(Hz,M1, label=r'$D=6, J_{\bigtriangledown}/J_{\bigtriangleup}=0.15$', marker="o")
    plt.scatter(Hz,M2, label=r'$D=6, J_{\bigtriangledown}/J_{\bigtriangleup}=0.50$', marker="d")
    plt.scatter(Hz,M3, label=r'$D=6, J_{\bigtriangledown}/J_{\bigtriangleup}=0.75$', marker="*")
    plt.scatter(Hz,M4, label=r'$D=6, J_{\bigtriangledown}/J_{\bigtriangleup}=1.00$', marker="^")
 
    plt.hlines([1.0/9.0], xmin=0.0, xmax=3.0, color="black", linestyles='dashed')
    plt.hlines([1.0/3.0], xmin=0.0, xmax=3.0, color="black", linestyles='dashed')
    #plt.scatter(x2,y2,c='#00cc00')

    # 軸名とタイトル
    plt.xlabel(r'$H_z$') #x軸名。Latex形式で書ける
    plt.ylabel(r'$M_z/M_{sat}$') #y軸名

    # 表示
    plt.grid()
    plt.rc('legend', fontsize=12)
    plt.legend()
    plt.savefig('BKH32.png')
    plt.show()
    plt.close()

if __name__ == '__main__':

    #plt.rcParams['font.family'] ='sans-serif'#使用するフォント
    plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
    plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
    plt.rcParams['font.size'] = 16 #フォントの大きさ
    plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
    #main()
    #bkh12_plot()
    bkh32_plot()
