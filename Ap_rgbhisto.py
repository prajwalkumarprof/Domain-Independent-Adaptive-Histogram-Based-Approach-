# Get all PDF documents in current directory
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image


i=0
path="./fruit-class1/"
#path="./leaf-class1/"


imageslist = []
for filename in os.listdir(path):
    if filename.endswith(".jpg"):
        imageslist.append(filename)
        print(filename)

imageslist.sort(key=str.lower)

#from PyPDF2 import PdfWriter, PdfReader


filename=imageslist[i]
image = cv2.imread( path+filename )

#cv2.imshow('Original', image)
#cv2.waitKey(5000)

img_gray = cv2. cvtColor(image, cv2.COLOR_BGR2GRAY)

#plt.imshow(img_gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

#image =PIL.Image.open("Images/"+imageslist[0] )

#img_gray = image.convert("L")

 
 

img_normalized = cv2.normalize(img_gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#cv2.imshow('Original', img_normalized)
#cv2.waitKey(1000)

 
#cv2.imwrite("7gray"+filename, img_gray)

#plt.hist(img_gray.ravel(), bins=range(256), fc='k', ec='k')
#plt.title('Mean')
#plt.xlabel("value")
#plt.ylabel("Frequency")
#plt.savefig("0Grayhisto.png")


#fig=plt.hist(img_normalized.ravel(), bins=range(50), fc='k', ec='k')
fig=plt.hist(img_normalized.ravel(),256,[0,1])
plt.title('Mean')
plt.xlabel("value")
plt.ylabel("Frequency")
#plt.savefig(path+filename+"normalizedHisto.png")



#plt.figure()
 
#f, axarr = plt.subplots(3,1,figsize =(10, 7),tight_layout = True)  
#axarr[0].imshow(image)
#axarr[1].imshow(img_gray)
#axarr[2].imshow(img_normalized)
#axarr[0].imshow(histo)

#for i, col in enumerate(['b', 'g', 'r']):
hist = cv2.calcHist([image],[0], None, [256], [0, 256])
plt.plot(hist, color = 'b')
plt.xlim([0, 256])
    
#plt.show()

hist = cv2.calcHist([image], [1], None, [256], [0, 256])
plt.plot(hist, color = 'g')
plt.xlim([0, 256])
    
#plt.show()

hist = cv2.calcHist([image], [2], None, [256], [0, 256])
plt.plot(hist, color = 'r')
plt.xlim([0, 256])
    
#plt.show()

 
#plt.show()
 

for filename in imageslist: 
    image = cv2.imread(path+filename)
    img_gray = cv2. cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_normalized = cv2.normalize(img_gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(img_normalized)

    ax2=plt.hist(img_normalized.ravel(),256,[0,1])
    plt.hist(img_normalized.ravel(),256,[0,1])
    plt.title('Mean')
    plt.xlabel("value")
    plt.ylabel("Frequency")
    plt.savefig(path+filename+"Histo-gray.png")
    plt.clf()

# end of combined image historgram gray
    plt.subplot(3, 1, 1)
    plt.subplots_adjust(left=0.1, bottom=0.09,right=0.9,top=0.9,wspace=0.4,hspace=0.4)
    plt.title("histogram of Blue")
    hist = cv2.calcHist([image],[0], None, [256], [0, 256])
    plt.plot(hist, color = 'blue')
    plt.xlim([0, 256])
    #plt.savefig(path+filename+"Histo-B.png") 
    
    plt.subplot(3, 1, 2)
    plt.title("histogram of  Red")
    hist2 = cv2.calcHist([image], [1], None, [256], [0, 256])
    plt.plot(hist2, color = 'red')
    plt.xlim([0, 256])
    #plt.savefig(path+filename+"Histo-G.png") 

    plt.subplot(3, 1, 3)
    plt.title("histogram of Green")
    hist3 = cv2.calcHist([image], [2], None, [256], [0, 256])
    plt.plot(hist3, color = 'green')
    plt.xlim([0, 256])
    plt.savefig(path+filename+"Histo-R.png");
    plt.clf()


    
