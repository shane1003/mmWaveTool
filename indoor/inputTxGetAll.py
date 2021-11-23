import time
import multiprocessing
from utils.models2 import CNNModel
from matplotlib import image, pyplot
import numpy as np
import math
import cv2
import tensorflow.compat.v1 as tf
import os
import time 
from socket import timeout
zoomLevel = 20
adjustValue = 19 / 2
SIDELENGTH = 15
IMAGE_SIZE= 40

os.chdir('./Indoor')
print(os.getcwd())

class inputTxGetAllPrediction:
    def __init__(self,Tx_X,Tx_Y, save_directory):
        self.checkpointPath = "Training_Model/model_checkpoint/2021-07-27-23-15-49/"
        self.Tx_X = Tx_X
        self.Tx_Y = Tx_Y
        self.save_directory = save_directory
        
    def prediction(self):        
        with tf.Graph().as_default():
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            model = CNNModel()
            #model = utils.models2.CNNModel()
#             print('\nEvaluating......')
            queues = []
            lamses = []
            allImages = []
            allDistances = []
            allX = []
            allY = []
            for i in range(0,1736,100):
                for j in range(0,574,100):
                    queue = multiprocessing.Queue()
                    queues.append(queue)
                    if i + 100>1736:
                        endI = 1736
                    else: 
                        endI = i+100
                    if j + 100>574:
                        endJ = 574
                    else: endJ = j+100    
                        
                    lams = LAMS(queue,self.Tx_X, self.Tx_Y,i, endI,j,endJ)
                    lamses.append(lams)
                    lams.start()
            for i, t in enumerate(lamses):
                result = queues[i].get()
                allImages.extend(result[0])
                allDistances.extend(result[1])
                allX.extend(result[2])
                allY.extend(result[3])
                
            input = np.reshape(allImages,(-1,IMAGE_SIZE,IMAGE_SIZE,1))
            distance = np.reshape(allDistances,(-1, 1))
            

            with tf.Session() as sess:
#                 print('Restoring from checkpoint...')
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(self.checkpointPath)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = \
                        ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
#                     print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')
                    return
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                
                prediction= sess.run(
                    [model.predictions],
                    feed_dict={model.input_ph: input,model.distance_ph: distance})
                a = (np.array(allX)).shape
                outPrediction = (np.array(prediction[0])).reshape(a[0],)
                
            imageArry = image.imread("imagedata/41-124.jpg")
            fig = pyplot.figure(figsize=(30,30))
            pyplot.imshow(imageArry)
            pyplot.scatter(allX,allY,100,outPrediction,alpha=0.5)# pyplot.scatter(x,y,20,PL,'filled')
            pyplot.set_cmap('jet')
            pyplot.colormaps()
            cb = pyplot.colorbar(fraction=0.018)
            font = {'family' : 'serif',
                  'color'  : 'black',
                  'weight' : 'normal',
                  'size'   : 30,
                 }
            cb.set_label('Path Loss',fontdict=font) 
            h = pyplot.scatter(self.Tx_X,self.Tx_Y,100,c = 'r',marker = 's',linewidths=2.5);
            pyplot.legend([h],('Transmitter (Tx)',),loc='upper center', fontsize= 'x-large',bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True) 
            fig.savefig(self.save_directory+"Tx_"+str(self.Tx_X)+"_"+str(self.Tx_Y)+"_PredictionResult.png")
            pyplot.pause(1)
            pyplot.close()
            
#             return outPrediction,allX,allY


class LAMS(multiprocessing.Process):

    def __init__(self,queue,Tx_X,Tx_Y, beginX,endX,beginY,endY):
        super().__init__()
        self.queue = queue
        imageArry = image.imread("imagedata/41-124gray.jpg")
        self.floorPlan = 255-imageArry
        self.Tx_X = Tx_X/20 #co 12.9
        self.Tx_Y = Tx_Y/20 #Ro 8.1
        self.halfL = int((SIDELENGTH - 1) / 2 * zoomLevel)
        self.beginX = beginX
        self.endX = endX
        self.beginY = beginY
        self.endY = endY

    #based on Rx position, to find the square, four position of square
    def findVertices(self):
        beginX = self.beginX
        endX = self.endX
        beginY = self.beginY
        endY = self.endY
        allImage = []
        allDistance = []
        allX = []
        allY = []
#         for x in range(200,250,10): #test
#             for y in range(500,550,10):
        n = 0
        for x in range(beginX,endX,10):
            for y in range(beginY,endY,10):
                distance = math.sqrt(math.pow(self.Tx_Y - (y/zoomLevel), 2) + math.pow(self.Tx_X - (x/zoomLevel), 2))
                Tx_Yf = int(self.Tx_Y * zoomLevel)
                Tx_Xf = int(self.Tx_X * zoomLevel)
#                 Rx_Yf = int(self.Rx_Y * zoomLevel)
#                 Rx_Xf = int(self.Rx_X * zoomLevel)
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
        #             print(position)
    #             return self.lamsImage(position), distance
                allImage.append(self.lamsImage(position))
                allDistance.append(distance)
                allX.append(x)
                allY.append(y)  
                n = n+1
#                 print("n:  ",n)
                if n%100==0:
                    print("n: ",n,"----",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        result =[]
        result.append(allImage)
        result.append(allDistance)
        result.append(allX)
        result.append(allY)
        
        self.queue.put(result)
    
    def lamsImage(self,position):
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
        # pyplot.imshow(bigImageStore[0:tl-1,0:tw-1],cmap='gray')
        # pyplot.title("Image Area")
        # pyplot.pause(1)
        # pyplot.close()

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

        return finalImage
    
    def run(self):
        self.findVertices()
