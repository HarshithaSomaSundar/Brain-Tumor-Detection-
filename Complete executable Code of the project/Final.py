import cv2
import numpy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import Tkinter
import tkMessageBox
from Tkinter import Tk
from Tkinter import *
from tkFileDialog import askopenfilename
from mainreader import ReadImage
import math
from scipy import ndimage
import random
import sys
import matplotlib.image as mpimg
import scipy.misc
from scipy import misc
from scipy.ndimage import imread
from PIL import Image
from numpy import *
import skimage
from skimage.morphology import watershed, disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
import ttk

filename = "unassigned"
top = Tkinter.Tk()
top.title('Hello, Tkinter!')
top.geometry('1000x1000')

limit=100
maxCycle=3000
D=50
lb=-5.12
RAND_MAX=2147483647
ub=5.12
runtime=1
ObjValSol=0
FitnessSol=0
neighbor=0
param2change=0
GlobalMin=0
run=0





def process():
    img = cv2.imread(filename)
    global blur
    blur= cv2.bilateralFilter(img,9,75,75)
    b=numpy.zeros([blur.shape[0],blur.shape[1],blur.shape[2]])
    b=numpy.copy(blur)
    cv2.imwrite('color_img.tif', b)
    
    '''s=cv2.imread(blur,0)
    imshow('s',s)'''
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

    

def FCM():
    class FCM():
            def __init__(self,imageName,n_clusters,epsilon=0.05,max_iter=-1):
                    self.m = 2
                    self.n_clusters = n_clusters
                    self.max_iter = max_iter
                    self.epsilon = epsilon

                    read = ReadImage(imageName)
                    print "THIS IS READ",read.getData()
                    self.X, self.numPixels = read.getData()
                    self.X = self.X.astype(np.float)
                    print "initial X:",self.X,self.X.shape

                    self.U = []
                    for i in range(self.numPixels):
                            index = i % n_clusters
                            l = [ 0 for j in range(n_clusters) ]
                            l[index] = 1
                            self.U.append(l)
                    self.U = np.array(self.U).astype(np.float)
                    self.U = self.U.reshape(self.numPixels,self.n_clusters)

                    #self.U_new = np.zeros((self.numPixels,self.n_clusters))
                    self.U_new = np.copy(self.U)
                    #self.h = np.zeros((self.n_clusters,self.numPixels))
                    
                    self.C = []
                    self.C = [1,85,255]
                    #self.C = [0,255]
                    #self.C = [150,200]
                    self.C = np.array(self.C).astype(np.float)
                    self.C = self.C.reshape(self.n_clusters,1)
                    print "initial C:\n",self.C,self.C.shape

                    Lambda = 2
                    self.hesitation = np.zeros((self.numPixels,self.n_clusters))
                    for i in range(self.numPixels):
                            for j in range(self.n_clusters):
                                    self.hesitation[i][j] = 1.0 - self.U[i][j] - ( (1 - self.U[i][j]) / (1 + (Lambda * self.U[i][j]) ) )

                    print self.hesitation



            def update_U(self):
                    for i in range(self.numPixels):
                            for j in range(self.n_clusters):
                                    sumation = 0
                                    for k in range(self.n_clusters):
                                            sumation += ( self.eucledian_dist(self.X[i],self.C[j]) / self.eucledian_dist(self.X[i],self.C[k]) ) ** (2 / (self.m-1) )
                                    self.U[i][j] = 1 / sumation

                    print "U : ",self.U

            def update_C(self):
                    for j in range(self.n_clusters):
                            num_sum = 0
                            den_sum = 0
                            for i in range(self.numPixels):
                                    num_sum += np.dot((self.U[i][j] ** self.m),self.X[i])
                                    den_sum += self.U[i][j] ** self.m
                            self.C[j] = np.divide(num_sum,den_sum)

                    print "C : ",self.C

            def calculate_h(self):
                    #self.h = np.zeros((self.n_clusters,self.numPixels))
                    h = np.zeros((self.n_clusters,self.numPixels))
                    u_rolled = np.zeros((self.numPixels ** 0.5,self.numPixels ** 0.5))
                    kernel = np.ones((5,5))
                    #kernel[2][2] = 4
                    #kernel[2][1] = kernel[1][2] = kernel[3][2] = kernel[2][3] = 2
                    #print self.U.transpose().shape,self.U.transpose()[0].shape
                    for i in range(self.n_clusters):
                            u_rolled = self.U.transpose()[i].reshape(self.numPixels ** 0.5,self.numPixels ** 0.5)
                            print u_rolled.shape
                            h_rolled = ndimage.convolve(u_rolled,kernel,mode='constant',cval=0.0)
                            #self.h[i] = h_temp.reshape(1,self.numPixels)
                            h[i] = h_rolled.reshape(1,self.numPixels)

                    h = h.transpose()
                    #self.h = self.h.transpose()
                    print "\n",h,h.shape
                    return h

            def compute_intuitionistic_U(self):
                    Lambda = 0.5
                    for i in range(self.numPixels):
                            for j in range(self.n_clusters):
                                    self.hesitation[i][j] = 1.0 - self.U[i][j] - ( (1 - self.U[i][j]) / (1 + (Lambda * self.U[i][j]) ) )
                    int_U = np.add(self.U,self.hesitation)
                    self.U = np.copy(int_U)


            def computeNew_U(self):
                    p = 1
                    q = 2
                    self.h = self.calculate_h()
                    for j in range(self.numPixels):
                            numer = 0.0
                            denom = 0.0
                            for i in range(self.n_clusters):
                                    numer = (self.U[j][i] ** p) * (self.h[j][i] ** q)
                                    for k in range(self.n_clusters):
                                            denom += (self.U[j][k] ** p) * (self.h[j][k] ** q)
                                    self.U_new[j][i] = numer/denom

                    self.U = np.copy(self.U_new)

            def calculate_DB_score(self):
                    sigma = np.zeros((3,1)).astype(np.float)
                    count = np.zeros((3,1))
                    result = np.zeros(shape=(self.numPixels,1))
                    result = np.argmax(self.U, axis = 1)
                    #self.Y = np.copy(self.X.astype(np.uint8))
                    #for i in xrange(self.numPixels):
                    #	self.Y[i] = self.C[self.result[i]].astype(np.int)

                    for i in range(self.n_clusters):
                            sigma[i] = 0
                            for j in range(self.numPixels):
                                    if result[j] == i:
                                            count[i] += 1
                                            sigma[i] += self.eucledian_dist(self.C[i],self.X[j])
                            sigma[i] = sigma[i]/count[i]

                    #print result,sigma,count

                    R_01 = (sigma[0] + sigma[1])/self.eucledian_dist(self.C[0],self.C[1])
                    R_02 = (sigma[0] + sigma[2])/self.eucledian_dist(self.C[0],self.C[2])
                    R_12 = (sigma[1] + sigma[2])/self.eucledian_dist(self.C[1],self.C[2])

                    D0 = max(R_01,R_02)
                    D1 = max(R_01,R_12)
                    D2 = max(R_02,R_12)

                    DB_score = (D0 + D1 + D2)/self.n_clusters
                    print "DB_score: ",DB_score

            def calculate_D_score(self):
                    sigma = np.zeros((3,1)).astype(np.float)
                    count = np.zeros((3,1))
                    result = np.zeros(shape=(self.numPixels,1))
                    result = np.argmax(self.U, axis = 1)

                    for i in range(self.n_clusters):
                            sigma[i] = 0
                            for j in range(self.numPixels):
                                    if result[j] == i:
                                            count[i] += 1
                                            sigma[i] += self.eucledian_dist(self.C[i],self.X[j])
                            sigma[i] = sigma[i]/count[i]

                    denom = max(sigma[0],sigma[1],sigma[2])
                    #print denom

                    d_01 = self.eucledian_dist(self.C[0],self.C[1])
                    d_02 = self.eucledian_dist(self.C[0],self.C[2])
                    d_12 = self.eucledian_dist(self.C[1],self.C[2])

                    D_01 = d_01/denom
                    D_02 = d_02/denom
                    D_12 = d_12/denom

                    D_score = min(D_01,D_02,D_12)
                    print "D_score: ",D_score



            def calculate_scores(self):
                    self.Vpc = 0.0
                    sum_j = 0.0
                    for j in range(self.numPixels):
                            sum_i = 0.0
                            for i in range(self.n_clusters):
                                    sum_i += self.U[j][i] ** 2
                                    #print "sum_i: ",sum_i
                            sum_j += sum_i
                            #print "sum_j: ",sum_j
                    self.Vpc = sum_j/self.numPixels
                    print "VPC: ",self.Vpc

                    self.Vpe = 0.0
                    sum_j = 0.0
                    for j in range(self.numPixels):
                            sum_i = 0
                            for j in range(self.n_clusters):
                                    sum_i += self.U[j][i] * math.log(self.U[j][i])
                            sum_j += sum_i
                    self.Vpe = -1 * (sum_j/self.numPixels)
                    print "VPE: ",self.Vpe

                    self.Vxb = 0.0
                    sum_j = 0.0
                    for j in range(self.numPixels):
                            sum_i = 0
                            for i in range(self.n_clusters):
                                    sum_i += self.U[j][i] * (self.eucledian_dist(self.X[j],self.C[i]) ** 2)
                            sum_j += sum_i
                    numer = 1 * sum_j
                    #dist = [ self.eucledian_dist(self.C[0],self.C[1]) ** 2, self.eucledian_dist(self.C[1],self.C[2]) ** 2 ,self.eucledian_dist(self.C[0],self.C[2]) ** 2]
                    #denom = self.numPixels * min(dist)
                    denom = self.numPixels * ( self.eucledian_dist(self.C[0],self.C[1]) ** 2 )
                    self.Vxb = numer/denom
                    print "VXB: ",self.Vxb

                    self.calculate_DB_score()
                    self.calculate_D_score()



            def eucledian_dist(self,a,b):
                    return np.linalg.norm(a-b)

            def form_clusters(self):
                    d = 100
                    if self.max_iter != -1:
                            for i in range(self.max_iter):
                                    print "loop : " , int(i)
                                    self.update_C()
                                    #temp = np.copy(self.U)
                                    temp = np.copy(self.U_new)
                                    self.update_U()
                                    self.compute_intuitionistic_U()
                                    self.computeNew_U()
                                    d = sum(abs(sum(self.U_new - temp)))
                                    print "THIS IS D"
                                    print d
                                    self.segmentImage(i)
                                    if d < self.epsilon:
                                            break
                    else:
                            i = 0
                            while d > self.epsilon:
                                    self.update_C()
                                    temp = np.copy(self.U)
                                    self.update_U()
                                    d = sum(abs(sum(self.U - temp)))
                                    print "loop : " , int(i)
                                    print d
                                    self.segmentImage(i)
                                    i += 1

            
            def segmentImage(self,image_count):
                    self.result = np.zeros(shape=(self.numPixels,1))
                    self.result = np.argmax(self.U, axis = 1)
                    global Y
                    self.Y = np.copy(self.X.astype(np.uint8))
                    #a = raw_input("press any key!")
                    for i in xrange(self.numPixels):
                            self.Y[i] = self.C[self.result[i]].astype(np.int)
                            #print self.Y[i]
                    self.Y = self.Y.reshape(self.numPixels ** 0.5,self.numPixels ** 0.5)
                    #self.Y = self.Y.reshape(75,75)
                    print "THIS IS WHAT WE WANT"
                    print self.Y,self.Y.shape,self.Y.dtype
                    cv2.imwrite('output_sifcm/' + str(image_count) + '.jpg' , self.Y)
                    image_count += 1
                    cv2.imshow('image',self.Y)
                    b=numpy.zeros([self.Y.shape[0],self.Y.shape[1]])
                    b=numpy.copy(self.Y)
                    cv2.imwrite('usedforabc.TIF', b)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()



    def main():
            #cluster = FCM('afcm1.jpg',2,0.01,300)
            #cluster = FCM('MRI.jpg',3,0.00005,100)
            cluster = FCM('color_img.tif',3,0.05,100)
            cluster.form_clusters()
            cluster.calculate_scores()

            #cluster.calculate_h()
            #cluster.show_result()

    if __name__ == '__main__':
            main()

def ABC():
    temp=asarray(Image.open('usedforabc.TIF'))
    x=temp.shape[0]
    y=temp.shape[1]
    temp.resize((x,y)) # a 2D array
    #print(temp)
    NP=temp.shape[0]
    FoodNumber=NP/2
    Foods=[[0 for x in range(0,D)]for x in range(0,FoodNumber)]
    foods=temp
    f=[0 for x in range(0,FoodNumber)]
    fitness=[0 for x in range(0,FoodNumber)]
    trial=[0 for x in range(0,FoodNumber)]
    prob=[0 for x in range(0,FoodNumber)]
    solution=[0 for x in range(0,D)]
    GlobalParams=[0 for x in range(0,D)]
    GlobalMins=[0 for x in range(0,runtime)]
    
    def CalculateFitness(fun):
        result=0.0
        if fun>=0:
            result=1/(fun+1)
        else:
            result=1+abs(fun)
        return result


    def MemoriseBestSource():
        global GlobalMin
        for i in range(0, FoodNumber):
            if(f[i]<GlobalMin):
                GlobalMin = f[i]
                for j in range(0,D):
                    GlobalParams[j]=Foods[i][j]

    def init(index):
        for j in range(0,D):
            r= round(random.uniform(0.0,1.0),6)#/(RAND_MAX+1)
            Foods[index][j]=(r*(ub-lb)+lb)
            solution[j]=Foods[index][j]
        f[index]=Rosenbrock(solution)
        fitness[index]=CalculateFitness(f[index])
        trial[index]=0

    def initial():
        for i in range(0,FoodNumber):
            init(i)
        GlobalMin=f[0]
        for i in range(0,D):
            GlobalParams[i]=Foods[0][i]

    def Rosenbrock(sol):
        top=0
        for j in range(0,D):
            top=top+sol[j]*sol[j]
        return top


    def SendEmployedBees():
        for i in range(0,FoodNumber):
            r= random.randint(0,49)#/(RAND_MAX+1)
            param2change=r
            r= random.randint(0,19)#/(RAND_MAX+1)
            neighbour=r
            while(neighbour==i):
                r=round(random.uniform(0.0,1.0),6)#/(RAND_MAX+1)
                neighbour=int(r*FoodNumber)
            for j in range(0,D):
                solution[j]=Foods[i][j]
            r= round(random.uniform(0.0,1.0),6)#/(RAND_MAX+1)
            solution[param2change]=Foods[i][param2change]+(Foods[i][param2change]-Foods[neighbour][param2change])*(r-0.5)*2
            if(solution[param2change]<lb):
                solution[param2change]=lb
            if(solution[param2change]>ub):
                solution[param2change]=ub
            ObjValSol=Rosenbrock(solution)
            FitnessSol=CalculateFitness(ObjValSol)
            if(FitnessSol>fitness[i]):
                trial[i]=0
                for j in range(0,D):
                    Foods[i][j]=solution[j]
                    f[i]=ObjValSol
                    fitness[i]=FitnessSol
            else:
                trial[i]=trial[i]+1

    def CalculateProbabilities():
        maxfit=float(fitness[0])
        for i in range(0,FoodNumber):
            if(fitness[i]>maxfit):
                maxfit=fitness[i]
        for i in range(0,FoodNumber):
            prob[i]=(0.9*(fitness[i]/maxfit))+0.1

    def SendOnlookerBees():
        t=0
        i=0
        while(t<FoodNumber):
            r= round(random.uniform(0.0,1.0),6)#/(RAND_MAX+1)
            if(r<prob[i]):
                t=t+1
            r= random.randint(0,49)#/(RAND_MAX+1)
            param2change=r
            r= random.randint(0,19)#/(RAND_MAX+1)
            neighbour=r
            while(neighbour==i):
                r= round(random.uniform(0.0,1.0),6)#/(RAND_MAX+1)
                neighbour=int(r*FoodNumber)
            for j in range(0,D):
                solution[j]=Foods[i][j]
            r= round(random.uniform(0.0,1.0),6)#(RAND_MAX+1)
            #print r
            solution[param2change]=Foods[i][param2change]+(Foods[i][param2change]-Foods[neighbour][param2change])*(r-0.5)*2
            if(solution[param2change]<lb):
                solution[param2change]=lb
            if(solution[param2change]>ub):
                solution[param2change]=ub
            ObjValSol=Rosenbrock(solution)
            FitnessSol=CalculateFitness(ObjValSol)
            if(FitnessSol>fitness[i]):
                trial[i]=0
                for j in range(0,D):
                    Foods[i][j]=solution[j]
                    f[i]=ObjValSol
                    fitness[i]=FitnessSol
            else:
                trial[i]=trial[i]+1
            i=i+1
            if(i==FoodNumber):
                i=0

    def SendScoutBees():
        materialindex=0
        for i in range(0,FoodNumber):
            if(trial[i]>trial[materialindex]):
                materialindex=i
        if(trial[materialindex]>=limit):
            init(materialindex)

    mean=0.0
    for run in range(0,runtime):
        initial()
        MemoriseBestSource()
        for iter in range(0,maxCycle):
            SendEmployedBees()
            CalculateProbabilities()
            SendOnlookerBees()
            MemoriseBestSource()
            SendScoutBees()
        for j in range(0,D):
            print("GlobalParama[j]",GlobalParams[j])
        print("run:",run+1,GlobalMin)
        GlobalMins[run]=GlobalMin
        mean=mean+GlobalMin
    mean=mean/runtime
    print("means of",runtime,"runs:",mean)
    b=numpy.zeros([foods.shape[0],foods.shape[1]])
    b=numpy.copy(foods)
    cv2.imwrite('useforwatershed.tif',b)
    plt.imshow(foods)
    plt.show()
    
def Watershed():
    image=skimage.io.imread('useforwatershed.TIF', as_grey=True)
  
    # denoise image
    denoised = rank.median(image, disk(10))
      
    # find continuous region (low gradient) --> markers
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndimage.label(markers)[0]
      
    #local gradient
    gradient = rank.gradient(denoised, disk(2))
      
    # process the watershed
    labels = watershed(gradient, markers)
      
    # display results
    fig, axes = plt.subplots(ncols=4, figsize=(8, 2.7))
    ax0, ax1, ax2, ax3 = axes
      
    ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax1.imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
    ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
    ax3.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
      
    for ax in axes:
        ax.axis('off')
      
    plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    plt.show()

    


def openFile():
    global filename
    filename = askopenfilename()
    root.destroy()


if __name__ == '__main__':

    root = Tkinter.Tk()
    ttk.Style().configure('green/black.TButton', foreground='blue', background='green')
    ttk.Style().configure('yellow.TButton', foreground='red', background='green')
    #C=Button(top, text='File Open', bg="red", command = openFile)
    #C.pack()
    top.configure(background="black")
    w = Label(top, fg='yellow',bg='black' ,text="BRAIN TUMOUR DETECTION", font=("Helvetica", 24))
    w.pack(padx=20,pady=20)
    C=ttk.Button(top, text='File Open',  style='yellow.TButton', command = openFile)
    C.pack(padx=50, pady=50)
    
    
    #B =Button(top, text ="Pre-Processing", width=200, height=7, command = helloCallBack)
    #B.pack()
    
    F =ttk.Button(top, text ="Pre-Processing", width=50, style='green/black.TButton', command = process)
    F.pack(padx=50, pady=50)
    
    B=ttk.Button(top, text ="Processing-FCM", width=50, style='green/black.TButton', command =FCM)
    B.pack(padx=30, pady=30)
    D1 =ttk.Button(top, text ="Processing-ABC", width=50, style='green/black.TButton', command =ABC)
    D1.pack(padx=30, pady=30)
    E=ttk.Button(top, text ="Post-Processing", width=50, style='green/black.TButton', command = Watershed)
    E.pack(padx=30, pady=30)

    #C.pack()
    #F.place(height=700, width=100)
    root.mainloop()
