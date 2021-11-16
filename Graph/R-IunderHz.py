# %matplotlib inline
# datをグラフ化し、使用する点を選択しながら、Icを決定していく
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ファイルの選択
typ = [('', '*')]
dir = r'C:\Users\###########\ExperimentalData'
files = filedialog.askopenfilenames(filetypes = typ, initialdir = dir) 
dataList = [["H", "Rmax", "Rmin", "Rfull", "Rhalf", "Rplusmax", "Rplusmin", "Rplus", "Rminusmax", "Rminusmin", "Rminus", "Icplus", "Icminus"]]

def interpolated_intercept(x, y1, y2):
    """Find the intercept of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """    

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R
    
    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    for idx in idxs:
        idx = idx[0]
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        plt.plot(x[idx], y1[idx], 'ms', ms=7, label='Nearest data-point method')
        plt.plot(xc, yc, 'co', ms=5, label='Nearest data-point, with linear interpolation')
        plt.text(x[idx], y1[idx], '{x}'.format(x=round(idx)), fontsize=10)
        
    plt.scatter(curI, RH)        
    plt.plot(x, y1, marker='o', mec='none', ms=4, lw=1, label='y1')
    plt.plot(x, y2, marker='o', mec='none', ms=4, lw=1, label='y2')
    plt.show()
        
    # 2点間を決める
    idx1 = int(input("Icplus:"))
    idx2 = int(input("Icminus:"))
    if idx2 == 9999:
        idx1 = int(input("Icplus:"))
        idx2 = int(input("Icminus:"))
        
        
    if idx1 == 0:
        xcplus = 0
    else:
        xcplus, ycplus = intercept((x[idx1], y1[idx1]),((x[idx1+1], y1[idx1+1])), ((x[idx1], y2[idx1])), ((x[idx1+1], y2[idx1+1])))
    if idx2 == 0:
        xcminus = 0
    else:
        xcminus, ycminus = intercept((x[idx2], y1[idx2]),((x[idx2+1], y1[idx2+1])), ((x[idx2], y2[idx2])), ((x[idx2+1], y2[idx2+1])))
    print("xc+:"+str(xcplus)+" xc-:"+str(xcminus))
    return xcplus, xcminus

    
# ファイルを読み込む
for f in files:
    data = pd.read_table(f, engine = 'python')
    curI = data.iloc[:, 0]
    RH = data.iloc[:, 1]
    data_len = len(curI)
    data_len_half = round(data_len / 2) # 全体の半分のデータ数
    target1 = 'ActualField_'  # targetより後ろを抽出
    idx1 = f.find(target1)
    target2 = 'mT___'
    idx2 = f.find(target2)
    H = f[idx1+len(target1):idx2]
    Rmax = sum(RH.iloc[0:10])/len(RH.iloc[0:10]) # 抵抗の最大値
#     Rmax = sum(RH.iloc[-30:])/len(RH.iloc[-30:]) # 抵抗の最大値
    Rmin = sum(RH.iloc[data_len_half:data_len_half+10])/len(RH.iloc[data_len_half:data_len_half+10]) # 抵抗の最小値
    Rfull = Rmax - Rmin # 抵抗の最大値と最小値の差
    Rhalf = (Rmax + Rmin) / 2 # 抵抗の最大値と最小値の間の値
    curIfirstHalf = curI[:data_len_half] # 電流データ前半
    curIlatterHalf = curI[data_len_half:] # 電流データ後半
    RHfirstHalf = RH[:data_len_half] # 抵抗データ前半
    Rplusmax = max(RHfirstHalf) # 抵抗データ前半の最大値
    Rplusmin = min(RHfirstHalf) # 抵抗データ前半の最小値
    Rplus = Rplusmax - Rplusmin # 抵抗データ前半の最大値と最小値の差
    RHlatterHalf = RH[data_len_half:]# 抵抗データ後半
    Rminusmax = max(RHlatterHalf) # 抵抗データ後半の最大値
    Rminusmin = min(RHlatterHalf) # 抵抗データ後半の最小値
    Rminus = Rminusmax - Rminusmin # 抵抗データ後半の最大値と最小値の差
#     #dataをグラフ化
    plt.hlines([Rmax, Rmin], curI.min(), curI.max(), colors="red")
    Icplus, Icminus = interpolated_intercept(curI, RH, [Rhalf] * data_len)
    dataList.append([float(H), Rmax, Rmin, Rfull, Rhalf, Rplusmax, Rplusmin, Rplus, Rminusmax, Rminusmin, Rminus, Icplus, Icminus])
np.savetxt(r'C:\Users\nittalab\Downloads\test.dat', np.array(dataList), delimiter='\t', fmt="%s")
