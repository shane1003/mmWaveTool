# -*- coding: utf-8 -*-

import math
import sympy
import pandas as pd
import numpy as np
from matplotlib import image
from sympy import *
from PIL import Image
import os

os.chdir('./suburban')
print(os.getcwd())

# Final code
datapath = "./" # data path
xls1name = "7Txlocations_addN7129S6,712N8S89,10NDuplicatedall5NSorderSouthAll.xlsx" # Tx excel name
xls2name = "allScenarios_timeSorted2_orderSouthAll.xlsx" # Rx excel name
imagePath1 = "./newStreetBuilding/" # map images path
imagePath2 = "_adjusted20+10.png" # final part of map name, without 'NSheet1'
resultPath = "Generated_Images/"#data path + folder of generated LAMS images
N = 41  # image size
# read excel
def readExcel():
    # read file content
    file1Content = pd.read_excel(datapath + xls1name, header=None)
    size = file1Content.shape
    sheetName = []

    file2Content = []
    for i in range(size[0]):
        name = file1Content.iloc[i][6]
        sheetName.append(name)
        file2Content.append(pd.read_excel(datapath + xls2name, sheet_name=eval(sheetName[i])))
    return file1Content, file2Content

def readTxContent(content):
    return TxContent.iloc[i][0], TxContent.iloc[i][1], TxContent.iloc[i][2], TxContent.iloc[i][3], TxContent.iloc[i][4], TxContent.iloc[i][5]

def readRxContent(content):
    size_S = RxContent[i].shape[1]
    if size_S == 8 or size_S == 10:
        return RxContent[i].iloc[:, 0],RxContent[i].iloc[:, 2],RxContent[i].iloc[:, 3],RxContent[i].iloc[:, 4], RxContent[i].iloc[:, 5],RxContent[i].iloc[:, 6],RxContent[i].iloc[:, 7]
    else:
        if size_S == 7 or size_S == 9:
            return RxContent[i].iloc[:, 0], RxContent[i].iloc[:, 1],RxContent[i].iloc[:, 2],RxContent[i].iloc[:, 3],RxContent[i].iloc[:, 4],RxContent[i].iloc[:, 5],RxContent[i].iloc[:, 6]

# longitude to plane coordinates
def calculateAsix(longti, latti):  
    Ra = 6378137
    Rb = 6356752.314245179
    d = math.pi / 180
    e1 = math.sqrt(1 - (Rb / Ra) * (Rb / Ra))
    longti = (longti) * d
    latti = latti * d
    x = Ra * (longti)
    y = Ra * math.log(
        math.tan(math.pi / 4 + latti / 2) * pow((1 - e1 * math.sin(latti)) / (1 + e1 * math.sin(latti)), e1 / 2))
    return x, y

def solveEquation(equation1, equation2):
    result = solve([equation1, equation2], [xm, ym])
    x = []
    y = []
    if len(result) == 2:
        return result[0][0], result[0][1], result[1][0], result[1][1]
    if len(result) == 4:
        if result[0] != result[2]:
            for i in range(0, 4, 2):
                x.append((result[i][0] + result[i + 1][0]) / 2)
                y.append((result[i][1] + result[i + 1][1]) / 2)
            return x[0], y[0], x[1], y[1]
        return result[0][0], result[0][1], result[1][0], result[1][1]


def bham(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx
    if dy == 0:
        q = np.zeros([dx + 1])
    else:
        q = (np.insert((np.diff(np.arange(math.floor(dx / 2), -dy * dx + math.floor(dx / 2) - 1, -dy) % dx) > 0) + 0, 0, 0)).reshape(dx + 1, 1)
    if steep:
        if y1 <= y2:
            y = np.arange(y1, y2 + 1).reshape((y2 + 1 - y1), -1)
        else:
            y = np.arange(y1, y2 - 1, -1).reshape(abs(y2 - 1 - y1), -1)
        if x1 <= x2:
            x = (x1 + np.cumsum(q)).reshape(q.shape[0], -1)
        else:
            x = x1 - np.cumsum(q).reshape(q.shape[0], -1)
    else:
        if x1 <= x2:
            x = np.arange(x1, x2 + 1).reshape((x2 + 1 - x1), -1)
        else:
            x = np.arange(x1, x2 - 1, -1).reshape(abs(x2 - 1 - x1), -1)
        if y1 <= y2:
            y = y1 + np.cumsum(q).reshape(q.shape[0], -1)
        else:
            y = y1 - np.cumsum(q).reshape(q.shape[0], -1)
    return x, y


def calc_Picture_Value_Vector_sameLength2_noSymmetric_return(x2, y2, xi1, yi1, N, imageArry):
    x_marka, y_marka = bham(int(round(x2)), int(round(y2)), int(round(xi1)), int(round(yi1)))
    Picture_Value_Vector = np.zeros([N])
    width_linea = x_marka.shape[0]
    length_linea = y_marka.shape[0]
    add = 0
    image_len = imageArry.shape[0]
    image_width = imageArry.shape[1]
    for i in range(1, N + 1):
        x_i = math.ceil(i * width_linea / N)
        y_i = math.ceil(i * length_linea / N)

        y_markai = y_marka[x_i - 1]
        x_markai = x_marka[y_i - 1]

        if y_markai > image_len or x_markai > image_width or y_markai < 1 or x_markai < 1:
            Picture_Value_Vector[i] = 0
            add = 1
        else:
            Picture_Value_Vector[i - 1] = imageArry[int(y_markai) - 1, int(x_markai) - 1]
    return Picture_Value_Vector, add


def calc_Picture_Value_Vector_sameLength2_LosNLos(x2, y2, xi1, yi1, N, imageArry):
    x_marka, y_marka = bham(int(round(x2)), int(round(y2)), int(round(xi1)), int(round(yi1)))
    width_linea = x_marka.shape[0]
    length_linea = y_marka.shape[0]
    Los = 1
    for i in range(1, N + 1):
        x_i = math.ceil(i * width_linea / N)
        y_i = math.ceil(i * length_linea / N)
        image_len = imageArry.shape[0]
        image_width = imageArry.shape[1]
        y_markai = y_marka[x_i - 1]
        x_markai = x_marka[y_i - 1]
        if y_markai > image_len or x_markai > image_width or y_markai < 1 or x_markai < 1:
            Los = 1
        else:
            if imageArry[int(y_markai), int(x_markai)] > 10:
                Los = 0
                break
    return Los

if __name__ == "__main__":
    TxContent, RxContent = readExcel()
#    TxContent = readTxContent()
#    RxContent = readRxContent()
    for i in range(len(TxContent)):
        #tx location, read from excel
        #map image left top corner and right bottom corner
        Tx_Long, Tx_Lat, TLT_Lat, TLT_Long, TRB_Lat, TRB_Long =readTxContent(TxContent)
        dataLength = RxContent[i].shape[0]
        #rss:id,latitude,longitude,4 rss value
        Idx,Lat, Long, RSS1, RSS2, RSS3, RSS4 = readRxContent(RxContent)
        Tx_power = 42
        max_PL = 158
        senName = eval(TxContent.iloc[i][6])
        NS_Sen = senName.split("Sheet")
        #different Tx power for different scenario
        if NS_Sen[0] == 'S':
            if eval(NS_Sen[1]) >= 9:
                Tx_power = 53
                max_PL = 160 
            # S1-8
            else:
                Tx_power = 39
                max_PL = 146

        RSS1 = Tx_power - RSS1
        RSS2 = Tx_power - RSS2
        RSS3 = Tx_power - RSS3
        RSS4 = Tx_power - RSS4
        RSS1 = RSS1.fillna(value=max_PL)
        RSS2 = RSS2.fillna(value=max_PL)
        RSS3 = RSS3.fillna(value=max_PL)
        RSS4 = RSS4.fillna(value=max_PL)
        print(imagePath1 + senName + imagePath2)
        imageArry = image.imread(imagePath1 + senName + imagePath2, 'GREY')

        image_len = imageArry.shape[0]
        image_width = imageArry.shape[1]
#minus 2 tiles
#here we use small image coordinates (more accurate)
#which take the right bottom tiles right bottom GPS coordinates
        tileSize2 = 2 * 256
        img_len_ = image_len - 2 * tileSize2
        img_wid_ = image_width - 2 * tileSize2
        Long_norm = (Long - TLT_Long) / (TRB_Long - TLT_Long) * img_wid_ + tileSize2
        Lat_norm = (Lat - TRB_Lat) / (TLT_Lat - TRB_Lat) * img_len_ + tileSize2

        Tx_Long_norm = (Tx_Long - TLT_Long) / (TRB_Long - TLT_Long) * img_wid_ + tileSize2
        Tx_Lat_norm = (Tx_Lat - TRB_Lat) / (TLT_Lat - TRB_Lat) * img_len_ + tileSize2

        oldx = 0
        oldy = 0
        num_outedge = 0
        #for each rss
        for j in range(dataLength):
            Rx_Long = Long[j]  # x1_
            Rx_Lat = Lat[j]  # y1_
            Rx_Long_norm = Long_norm[j]  # x1
            Rx_Lat_norm = Lat_norm[j]  # y1
            distance = math.sqrt((Tx_Long_norm - Rx_Long_norm) ** 2 + (Rx_Lat_norm - Tx_Lat_norm) ** 2)

            ratio_extend = 10
            x1_extend = ((1 + ratio_extend) * Rx_Long_norm - Tx_Long_norm) / ratio_extend
            y1_extend = ((1 + ratio_extend) * Rx_Lat_norm - Tx_Lat_norm) / ratio_extend

            Rx_Long_norm = x1_extend
            Rx_Lat_norm = y1_extend
            distance = distance * 1.1  

            Tx_x, Tx_y = calculateAsix(Tx_Long, Tx_Lat)
            Rx_x, Rx_y = calculateAsix(Rx_Long, Rx_Lat)
            distance_TR = math.sqrt((Tx_x - Rx_x) ** 2 + (Tx_y - Rx_y) ** 2)
            #choose 30<d<600
            if distance_TR > 600 or distance_TR < 30:
                continue
            if ~(oldx == Rx_Long_norm and oldy == Rx_Lat_norm):
                lams_image = np.zeros([N, N])
                distance_A05 = distance / 2

                A = Tx_Lat_norm - Rx_Lat_norm
                B = Rx_Long_norm - Tx_Long_norm
                k_A = -A / B
                c_A = Tx_Long_norm * Rx_Lat_norm - Rx_Long_norm * Tx_Lat_norm
                if j != 1:
                    current_index = j
                    for lastSearch in range(15):
                        last_index = j - lastSearch
                        if last_index <= 0:
                            last_index = j
                            for currentSearch in range(15):
                                current_index = j + currentSearch
                                sum = Long[j] - Long[current_index] + Lat[j] - Lat[current_index]
                                if sum != 0:
                                    break
                        sum = (Long[j] - Long[last_index]) + (Lat[j] - Lat[last_index])
                        if sum != 0:
                            break

                else:
                    if j == 1:
                        last_index = j
                        for currentSearch in range(15):
                            current_index = j + currentSearch
                            sum = Long[j] - Long[current_index] + Lat[j] - Lat[current_index]
                            if sum != 0:
                                break
                current_Rx_Long = Long[current_index]
                current_Rx_Lat = Lat[current_index]
                current_Rx_Long_norm = Long_norm[current_index]
                current_Rx_Lat_norm = Lat_norm[current_index]

                last_Rx_Long = Long[last_index]
                last_Rx_Lat = Lat[last_index]
                last_Rx_Long_norm = Long_norm[last_index]
                last_Rx_Lat_norm = Lat_norm[last_index]

                Llx1_ = 2 * current_Rx_Long - last_Rx_Long
                Lly1_ = 2 * current_Rx_Lat - last_Rx_Lat
                len_C = math.sqrt((Tx_Long - Llx1_) ** 2 + (Tx_Lat - Lly1_) ** 2)
                len_t = math.sqrt((current_Rx_Long - Llx1_) ** 2 + (current_Rx_Lat - Lly1_) ** 2)

                len_ll = math.sqrt((current_Rx_Long - Tx_Long) ** 2 + (current_Rx_Lat - Tx_Lat) ** 2)

                angle_C_TCLl = math.acos((len_t ** 2 + len_ll ** 2 - len_C ** 2) / (2 * len_t * len_ll)) * 180 / math.pi
                A_R1 = 52.5 + angle_C_TCLl
                A_R2 = 77.5 + angle_C_TCLl
                A_R3 = 102.5 + angle_C_TCLl
                A_R4 = 127.5 + angle_C_TCLl
                xV = (Rx_Long_norm + Tx_Long_norm) / 2
                yV = (Rx_Lat_norm + Tx_Lat_norm) / 2
                CMN = B * xV - A * yV
                xm = Symbol('xm')
                ym = Symbol('ym')
                equ1 = distance_A05 ** 2 - (xV - xm) ** 2 - (yV - ym) ** 2 
                equ2 = (xm - Rx_Long_norm) ** 2+(ym - Rx_Lat_norm) ** 2 - (distance_A05 ** 2) * 2 
                xm_a, ym_a, xm_b, ym_b = solveEquation(equation1=equ1, equation2=equ2)
                if (Rx_Long_norm - Tx_Long_norm) >= 0:
                    if ym_a < ym_b:
                        xm_a, xm_b = xm_b, xm_a
                        ym_a, ym_b = ym_b, ym_a
                else:
                    if ym_a >= ym_b:
                        xm_a, xm_b = xm_b, xm_a
                        ym_a, ym_b = ym_b, ym_a
                xmc = 0; xnc = 0;ymc = 0;ync = 0
                if xm_a > image_width:
                    xm_a = image_width
                    xmc = 1
                    ym_a = (-CMN + B * xm_a) / A
                if xm_b > image_width:
                    xm_b = image_width
                    ym_b = (-CMN + B * xm_b) / A
                    xnc = 1
                if xm_a < 1:
                    xm_a = 1
                    xmc = 1
                    ym_a = -CMN / A
                if xm_b < 1:
                    xm_b = 1
                    xnc = 1
                    ym_b = -CMN / A
                if ym_a > image_len:
                    ym_a = image_len
                    xm_a = (A * ym_a + CMN) / B
                    ymc = 1
                if ym_b > image_len:
                    ym_b = image_len
                    xm_b = (A * ym_b + CMN) / B
                    ync = 1
                if ym_a < 1:
                    ym_a = 1
                    ymc = 1
                    xm_a = CMN / B
                if ym_b < 1:
                    ym_b = 1
                    xm_b = CMN / B
                    ync = 1
                if ymc or ync or xmc or xnc:
                    num_outedge = num_outedge + 1
                    continue

                add_0_value = 'NoAppending'
                x = Symbol('x'); y = Symbol('y') 
                equation3 = B * (x - Tx_Long_norm) - A * (y - Tx_Lat_norm)
                equation4 = B * (x - Rx_Long_norm) - A * (y - Rx_Lat_norm)
                equation5 = A * (x - xm_a) + B * (y - ym_a)
                equation6 = A * (x - xm_b) + B * (y - ym_b)
                equation = [equation3, equation4, equation5, equation6]
                x_p=[];y_p=[]
                for e in range(2):
                    for k in range(2,4):
                        rp = solve([equation[e],equation[k]],[x,y])
                        x_p.append(rp[x])
                        y_p.append(rp[y])

                for ai in range(N):
                    x13i = x_p[0] - (ai) * ((x_p[0] - x_p[2]) / (N - 1))
                    y13i = y_p[0] - (ai) * ((y_p[0] - y_p[2]) / (N - 1))
                    x24i = x_p[1] - (ai) * ((x_p[1] - x_p[3]) / (N - 1))
                    y24i = y_p[1] - (ai) * ((y_p[1] - y_p[3]) / (N - 1))

                    Picture_Value_Vector_ai, add = calc_Picture_Value_Vector_sameLength2_noSymmetric_return(x13i, y13i,x24i, y24i,N,imageArry)
                    if add == 1:
                        add_0_value = 'Append0'
                    lams_image[:, ai] = Picture_Value_Vector_ai.squeeze()

                oldx = Rx_Long_norm
                oldy = Rx_Lat_norm
                old_image = lams_image  # New_VS
            else:
                lams_image = old_image

            em = '0'
            if np.max(np.max(lams_image)) == 0:
                em = '1'

            isLoS = calc_Picture_Value_Vector_sameLength2_LosNLos(Tx_Long_norm, Tx_Lat_norm, Rx_Long_norm, Rx_Lat_norm,round(distance_TR), imageArry)
            if isLoS:
                S_isLoS = '1'
            else:
                S_isLoS = '0'
            imageName = senName + '_' + str(j + 1) + '_' + str(round(Rx_Long, 7)) + '_' + str(
                round(Rx_Lat, 7)) + '_' + str(round(RSS1[j], 4)) + '_' + str(round(RSS2[j], 4)) + '_' + str(round(RSS3[j], 4)) + '_' + str(
                round(RSS4[j], 4)) + '_' + str(round(distance_TR, 6))+ '_' + str(round(A_R1, 3)) + '_' + str(
                round(A_R2, 3)) + '_' + str(round(A_R3, 3)) + '_' + str(round(A_R4, 3)) +  '_' + S_isLoS + '_' + em + '_' + add_0_value + '.png'
            print(imageName)

            final_image = np.uint8(lams_image)
            lamsPath = datapath + resultPath + senName + '/'
            if not os.path.exists(lamsPath):
                os.makedirs(lamsPath)
            im = Image.fromarray(final_image)
            im.save(lamsPath+ imageName)