import numpy as np
 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
 
# Function to calculate Chi-distance

def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	# return the chi-squared distance
	return d
 
# main function
if __name__== "__main__":
    a = [1, 0, 0, 5, 45, 23]
    b = [67, 90, 0, 79, 24, 98]
 
  #  result = chi2_distance(a, b)
#print("The Chi-square distance is :", result)


i=0

#path="./testangle2/"
path="./test2distortion/"
#path="./fruit-class4/"
#path="./leaf-class5/"


OPENCV_METHODS = (
	("Correlation", cv2.HISTCMP_CORREL),
	("Chi-Squared", cv2.HISTCMP_CHISQR),
	("Intersection", cv2.HISTCMP_INTERSECT),
	("Hellinger", cv2.HISTCMP_BHATTACHARYYA))




def callhistogram(path,imageslist):
    imageslist.sort(key=str.lower)
    for filename in imageslist: 
        image = cv2.imread(path+"/"+filename)
        img_gray = cv2. cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray[img_gray > 250] =0
        img_normalizedgray = cv2.normalize(img_gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
       # print(img_gray)

        b,g,r = cv2.split(image)
        img_normalized_b = cv2.normalize(b,   0, 256, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_normalized_r = cv2.normalize(r,   0, 256, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_normalized_g = cv2.normalize(g,   0, 256, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

      #  fig, (ax1, ax2) = plt.subplots(2)
      #  ax1.imshow(img_normalized)
       # ax2=plt.hist(img_normalized.ravel(),256,[0,1])

        axisgray=plt.hist(img_normalizedgray[img_normalizedgray > 0],256,[0,1])
        plt.title('GRAY HISTOGRAM ')
        plt.xlabel("value")
        plt.ylabel("Frequency")
        plt.savefig(path+"/"+filename+"Histo-Zgray.png")
        plt.clf()

    # end of combined image historgram gray
       # plt.subplot(3, 1, 1)
      #  plt.subplots_adjust(left=0.1, bottom=0.09,right=0.9,top=0.9,wspace=0.4,hspace=0.4)
       

        plt.title("histogram of Blue")
        axisB=plt.hist(img_normalized_b.ravel(),256,[0,1]) 
        
        plt.xlabel("value")
        plt.ylabel("Frequency")
        plt.savefig(path+"/"+filename+"Histo-B.png") 
        plt.clf()

      #  plt.subplot(3, 1, 2)
        plt.title("histogram of  Red")
        axisr= plt.hist(img_normalized_r.ravel(),256,[0,1]) 
        plt.savefig(path+"/"+filename+"Histo-R.png") 
        plt.clf()

       # plt.subplot(3, 1, 3)
        plt.title("histogram of Green")
        axisg=plt.hist(img_normalized_g.ravel(),256,[0,1]) 
        plt.savefig(path+"/"+filename+"Histo-G.png");
        plt.clf()
       # print(axisg[0])



        print (path+filename)
        print(" Chi-square distance :")
        a=[i for i in axisg[0] if i != 0]
        b=[i for i in axisgray[0] if i != 0]

        resultG = chi2_distance(a, b)
        print(" D-G The Chi-square distance is :", resultG)


      #  print(" D-R  Chi-square distance :")
        a=[i for i in axisr[0] if i != 0]
        b=[i for i in axisgray[0] if i != 0]

        resultR = chi2_distance(a, b)
        print(" D-R The Chi-square distance is :", resultR)   

       # print(" D-B  Chi-square distance :")
        a=[i for i in axisB[0] if i != 0]
        b=[i for i in axisgray[0] if i != 0]
        resultB = chi2_distance(a, b)
        print(" D-B The Chi-square distance is :", resultB) 


        print("  Features distance :" +path+filename)
        print(" Peak value -Green distance :" )
        print(axisg[0].max())
        print(" Peak value -Red distance :" )
        print(axisr[0].max())
        print(" Peak value -Blue distance :" )
        print(axisB[0].max())
        print(" Peak value -Gray distance :" )
        print(axisgray[0].max())


        alen = len(axisg[0])
        print("ARRAY LENTH1 ")
        print(alen)





        axisgP1 =np.array_split(axisg[0], 2)
        axisrP1 =np.array_split(axisr[0], 2)
        axisBP1  =np.array_split(axisB[0], 2)
        axisgrayP1,axisgrayP2=np.array_split(axisgray, 2)

        alen = len(axisgP1[0])
        print("ARRAY LENTH part a")
        print(alen)

        alen = len(axisgP1[1])
        print("ARRAY LENTH part b")
        print(alen)

        b=[i for i in axisgrayP1[0] if i != 0]

        print(" Chi-square distance :" +path+filename)
        a=[i for i in axisgP1[0] if i != 0]
        

       # resultG = cv2.compareHist(axisg, axisgray,  cv2.HISTCMP_CHISQR)
        resultG1 = chi2_distance(a, b)
        print(" P1 - D-G The Chi-square distance is :", resultG1)


      #  print(" D-R  Chi-square distance :")
        a=[i for i in axisrP1[0] if i != 0]
        
        resultR1 = chi2_distance(a, b)
       # resultR=  cv2.compareHist(axisr, axisgray,  cv2.HISTCMP_CHISQR)
        print("P1 -  D-R The Chi-square distance is :", resultR1)   

       # print(" D-B  Chi-square distance :")
        a=[i for i in axisBP1[0] if i != 0]
        resultB1 = chi2_distance(a, b)
        #resultB=  cv2.compareHist(axisB, axisgray,  cv2.HISTCMP_CHISQR)
        print("P1 - D-B The Chi-square distance is :", resultB1) 


            #--------------------------

        b=[i for i in axisgrayP1[1] if i != 0]

        print(" Chi-square distance :" +path+filename)
        a=[i for i in axisgP1[1] if i != 0]
        

        # resultG = cv2.compareHist(axisg, axisgray,  cv2.HISTCMP_CHISQR)
        resultG2 = chi2_distance(a, b)
        print(" P2 - D-G The Chi-square distance is :", resultG2)


       #  print(" D-R  Chi-square distance :")
        a=[i for i in axisrP1[1] if i != 0]
        
        resultR2 = chi2_distance(a, b)
        # resultR=  cv2.compareHist(axisr, axisgray,  cv2.HISTCMP_CHISQR)
        print("P2 -  D-R The Chi-square distance is :", resultR2)   

       # print(" D-B  Chi-square distance :")
        a=[i for i in axisBP1[1] if i != 0]
        resultB2= chi2_distance(a, b)
        #resultB=  cv2.compareHist(axisB, axisgray,  cv2.HISTCMP_CHISQR)
        print("P2 - D-B The Chi-square distance is :", resultB2) 


      #  return resultG 


imageslist = []

for filename in os.listdir(path):
    if filename.endswith(".jpg"):
        imageslist.append(filename)


        
callhistogram(path,imageslist)


#







#Directorylist = []




#for pathb in os.listdir("."):
#    Directorylist.append(pathb)

#Directorylist.sort(key=str.lower)

#for pathb in Directorylist:
  
  #  print(pathb)
 #   imageslist = []
 #   if os.path.isdir(pathb):
 #    for filename in os.listdir(pathb):
#          if filename.endswith(".jpg"):
#            imageslist.append(filename)
   #callhistogram(pathb,imageslist)
