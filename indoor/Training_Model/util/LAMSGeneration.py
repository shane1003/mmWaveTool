import time
from matplotlib import image, pyplot
import numpy as np
import math
import tensorflow as tf
import os
import cv2
import time 
from socket import timeout
import pandas as pd
from PIL import Image
zoomLevel = 20
adjustValue = 19 / 2
SIDELENGTH = 15
IMAGE_SIZE= 40

class LAMS():

    def __init__(self,imageArry,fileData,fileID,saveDir):
        self.floorPlan = 255-imageArry
#         self.Tx_X = Tx_X/20 #co 12.9
#         self.Tx_Y = Tx_Y/20 #Ro 8.1
        self.Tx_X = 12.9 #co 
        self.Tx_Y = 8.1 #Ro 
        self.halfL = int((SIDELENGTH - 1) / 2 * zoomLevel)
        self.fileData = fileData
        self.fileID=fileID
        self.saveDir = saveDir

    #based on Rx position, to find the square, four position of square
    def findVertices(self,):
        size = self.fileData.shape
        startX = size[0]
        startY = size[1]
        
        allImage = []
        allDistance = []
        allX = []
        allY = []
        n = 0
        
        print("LAMS Generation start")
        for x in range(0,startX):
            for y in range(0,startY):
                if pd.isnull(self.fileID.iloc[x][y]):
                    continue
                beamId = int(self.fileID.iloc[x][y])
#                 print(beamId)
                label = self.fileData.iloc[x][y]
                Rx_Xf=(x+0.7)*zoomLevel-adjustValue;
                Rx_Yf=(y+5.3+y/69)*zoomLevel-adjustValue;
                distance = math.sqrt(math.pow(self.Tx_Y - (Rx_Yf/zoomLevel), 2) + math.pow(self.Tx_X - (Rx_Xf/zoomLevel), 2))

                Tx_Yf = int(self.Tx_Y * zoomLevel)
                Tx_Xf = int(self.Tx_X * zoomLevel)
                
                position = np.array(np.empty([4,2]))
                if y == Tx_Yf:
                    position= [[Tx_Yf - self.halfL, Tx_Xf],[Tx_Yf + self.halfL, Tx_Xf],[y + self.halfL, x],[
                                          y - self.halfL, x]]
                else:
                    if x == Tx_Xf:
                        position= [[Tx_Yf, Tx_Xf + self.halfL],[Tx_Yf, Tx_Xf - self.halfL],[y, x - self.halfL],[
                                          y, x + self.halfL]]
                    else:
                        k = (y - Tx_Yf) / (x - Tx_Xf)
                        newK = -1 / k
                        if abs(x - Tx_Xf) < abs(y - Tx_Yf):
                            if newK > 0:
                                addI = math.ceil(newK * (self.halfL) + y)
                                minI = math.floor(newK * (-self.halfL) + y)
                                caddI = math.ceil(newK * (self.halfL) + Tx_Yf)
                                cminI = math.floor(newK * (-self.halfL) + Tx_Yf)
                            else:
                                addI = math.floor(newK * (self.halfL) + y)
                                minI = math.ceil(newK * (-self.halfL) + y)
                                caddI = math.floor(newK * (self.halfL) + Tx_Yf)
                                cminI = math.ceil(newK * (-self.halfL) + Tx_Yf)
                            if y < Tx_Yf:
                                position= [[caddI, Tx_Xf + self.halfL],[cminI, Tx_Xf - self.halfL],[minI, x - self.halfL],[
                                                      addI, x + self.halfL]]

                            else:
                                position= [[cminI, Tx_Xf - self.halfL],[caddI, Tx_Xf + self.halfL],[addI, x + self.halfL],[
                                                      minI, x - self.halfL]]

                        else:
                            if newK > 0:
                                addJ = math.ceil(self.halfL / newK + x)
                                minJ = math.floor(-self.halfL / newK + x)
                                caddJ = math.ceil(self.halfL / newK + Tx_Xf)
                                cminJ = math.floor(-self.halfL / newK + Tx_Xf)
                            else:
                                addJ = math.floor(self.halfL / newK + x)
                                minJ = math.ceil(-self.halfL / newK + x)
                                caddJ = math.floor(self.halfL / newK + Tx_Xf)
                                cminJ = math.ceil(-self.halfL / newK + Tx_Xf)
                            if x < Tx_Xf:
                                position= [[Tx_Yf - self.halfL, cminJ],[Tx_Yf + self.halfL, caddJ],[y + self.halfL, addJ],[
                                                      y - self.halfL, minJ]]
                            else:
                                position= [[Tx_Yf + self.halfL, caddJ],[Tx_Yf - self.halfL, cminJ],[y - self.halfL, minJ],[
                                                      y + self.halfL, addJ]]

                self.lamsImage(position,beamId,x,y,label,distance)
        print("LAMS Generation end")
    
    def lamsImage(self,position,beamId,x,y,label,distance):
        dxA = position[0][1] - position[1][1] # j
        dyA = position[0][0]  - position[1][0] # i
        dxB = position[2][1]  - position[1][1] # j
        dyB = position[2][0] - position[1][0] # i
        bigLength = math.ceil(math.sqrt(pow(dxA,2)+pow(dyA,2)))+1
        bigWidth =math.ceil(math.sqrt(pow(dxB,2)+pow(dyB,2)))+1
#         print("Big size:",bigLength,bigWidth)
        bigImageStore = np.zeros((bigLength,bigWidth))
        m = 0;n = 0
        tw = 0;tl=0
        if abs(dxA) > abs(dyA):
            for jA in np.linspace(position[1][1],position[0][1],num = abs(dxA)+1):
                jC = position[2][1] + (jA - position[1][1])
                k = dyA / dxA
                iA = math.floor(k * (jA - position[1][1]) + position[1][0])
                iC = math.floor(k * (jC - position[2][1]) + position[2][0])
                if abs(dxB) > abs(dyB):
                    for jm in np.linspace(jA, jC,num = int(abs(jA-jC)+1)):
                        kT = dyB / dxB
                        im = math.ceil(kT * (jm - jA) + iA)
                        if im > 28.7 * zoomLevel or im < 1 or jm > 86.8 * zoomLevel or jm < 1 :
                            bigImageStore[m, n] = 0
                        else:
                            bigImageStore[m, n]= self.floorPlan[math.ceil(im)-1, math.ceil(jm)-1]
                        n = n + 1
                    tw = n
                    n = 0
                    m = m + 1
                else:
                    for im in np.linspace(iA,iC,num = int(abs(iA-iC)+1)):
                        if dxB==0:
                            jm = jA
                        else:
                            kT = dyB / dxB
                            jm = math.floor((im - iA) / kT + jA)
                        if im > 28.7 * zoomLevel or im < 1 or jm > 86.8 * zoomLevel or jm < 1:
                            bigImageStore[m, n] = 0
                        else:
                            bigImageStore[m, n] = self.floorPlan[math.ceil(im)-1, math.ceil(jm)-1]
                        n = n + 1
                    tw = n
                    n = 0
                    m = m + 1
        else:
            for iA in np.linspace(position[1][0], position[0][0],num = abs(dyA)+1):
                iC = position[2][0] + (iA - position[1][0])
                if dxA == 0:
                    jA = position[1][1]
                    jC = position[2][1]
                else:
                    k = dyA / dxA
                    jA = math.floor((iA - position[1][0]) / k + position[1][1])
                    jC = math.floor((iC - position[2][0]) / k + position[2][1])
                if abs(dxB)>abs(dyB):
                    for jm in np.linspace(jA , jC,num = int(abs(jA -jC)+1)):
                        kT = dyB/dxB
                        im = math.ceil(kT*(jm-jA)+iA)
                        if im>28.7*zoomLevel or im<1 or jm>86.8*zoomLevel or jm<1 :
                            # if n >= bigWidth:
                            #     continue
                            bigImageStore[m,n] = 0
                        else:
                            bigImageStore[m,n] = self.floorPlan[math.ceil(im)-1, math.ceil(jm)-1]
                        n = n+1
                    tw = n
                    n = 0
                    m=m+1
                else:
                    for im in np.linspace(iA , iC,num = int(abs(iA-iC)+1)):
                        if dxB==0:
                            jm = jA
                        else:
                            kT = dyB/dxB
                            jm = math.floor((im-iA)/kT+jA)
                        if im>28.7*zoomLevel or im<1 or jm>86.8*zoomLevel or jm<1 :
                            bigImageStore[m,n] = 0
                        else:
                            bigImageStore[m,n] = self.floorPlan[math.ceil(im)-1, math.ceil(jm)-1]
                        n = n+1
                    tw = n
                    n = 0
                    m=m+1

        tl = m

        finalImage = np.zeros((IMAGE_SIZE,IMAGE_SIZE))
        m=0;n=0
        for it in np.linspace(1,tl,IMAGE_SIZE):
            it = math.floor(it)
            for jt in np.linspace(1,tw,IMAGE_SIZE):
                jt = math.floor(jt)
                finalImage[m,n] = bigImageStore[it-1,jt-1]
                n = n+1
            n=0
            m=m+1
            
        final_image = np.uint8(finalImage)
        lamsPath = self.saveDir+str(beamId)+'/'
        if not os.path.exists(lamsPath):
                os.makedirs(lamsPath)
        outputPath = lamsPath+str(beamId)+'_'+str(x+1)+'_'+str(y+1)+'_'+str(label)+'_'+str(round(distance, 2))+'.jpg'
        im = Image.fromarray(final_image)
        im.save(outputPath)
