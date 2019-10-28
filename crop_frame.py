import cv2
import os
import glob

def facecrop(imagePattern):
    facedata = "C:/Users/mihaela/Downloads/deepgaze-master/deepgaze-master/etc/xml/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)
    imgList=glob.glob(imagePattern)
    print(len(imgList))
    if len(imgList)<=0:
        print ('No Images Found')
        return
    #print (imgList)
    for image in imgList:
        print (image)
        img = cv2.imread(image)
    
        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)
    
        faces = cascade.detectMultiScale(miniframe)
    
        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
    
            sub_face = img[y:y+h, x:x+w]
            fname, ext = os.path.splitext(image)
            cv2.imwrite(fname+"_cropped"+ext, sub_face)
    
        #return
def facecropfile(image):
    facedata = "C:/Users/mihaela/Downloads/deepgaze-master/deepgaze-master/etc/xml/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)
    #imgList=glob.glob(imagePattern)
    #print(len(imgList))
    #if len(imgList)<=0:
     #   print ('No Images Found')
      #  return
    #print (imgList)

    img = cv2.imread(image)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        fname, ext = os.path.splitext(image)
        cv2.imwrite(fname+"_cropped_"+ext, sub_face)




facecropfile('headPose.jpg')
#facecrop("hpdb/*.png")
