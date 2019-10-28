#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import glob
import numpy
import scipy.io as sio

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

sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
roll_mat=[]
pitch_mat=[]
yaw_mat=[]
# Load the weights from the configuration folders
my_head_pose_estimator.load_roll_variables(os.path.realpath("C:/Users/mihaela/Downloads/deepgaze-master/deepgaze-master/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_pitch_variables(os.path.realpath("C:/Users/mihaela/Downloads\deepgaze-master/deepgaze-master/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_yaw_variables(os.path.realpath("C:/Users/mihaela/Downloads/deepgaze-master/deepgaze-master/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))
facecrop("test_andreea/*.jpg")
imagePattern='test_andreea/*_cropped.jpg'

imgList=glob.glob(imagePattern)

for i in imgList:
    #file_name = str(i) + "_cropped.png"
    file_name = str(i)
    print("Processing image ..... " + file_name)
    image = cv2.imread(file_name) #Read the image with OpenCV
    # Get the angles for roll, pitch and yaw
    roll = my_head_pose_estimator.return_roll(image)  # Evaluate the roll angle using a CNN
    pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
    yaw = my_head_pose_estimator.return_yaw(image)# Evaluate the yaw angle using a CNN
    #rpy.append(int(i[13:16]))
    pitch_mat.append(float(pitch[0,0,0]))
    yaw_mat.append(float(yaw[0,0,0]))
    roll_mat.append(float(roll[0,0,0]))
    print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
    print("")

roll_mat=numpy.asarray(roll_mat)
pitch_mat=numpy.asarray(pitch_mat)
yaw_mat=numpy.asarray(yaw_mat)
#rpy = rpy.reshape((len(imgList), 3))
sio.savemat('roll_subj3.mat', mdict={'roll_cnn_3': roll_mat})
sio.savemat('pitch_subj3.mat', mdict={'pitch_cnn_3': pitch_mat})
sio.savemat('yaw_subj3.mat', mdict={'yaw_cnn_3': yaw_mat})
#print(rpy)
#print(len(rpy))
#print(len(imgList))
