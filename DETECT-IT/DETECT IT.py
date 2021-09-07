#TAGS


from colorama import Fore, Back, Style
print('''
  ____ _____ _____ _____ ____ _____    ___ _____ 
 / _  |____ |_   _|____ |___ |_   _|  |_ _|_   _|
| | | | |_  | | |   |_  |   | || |_____| |  | |  
| |_| |___| | | |  ___| |___| || |_____| |  | |  
 \____|_____| |_| |_____|____/ |_|    |___| |_|  v0.2
                                                    -OJASWA RHX

''')







#Installing OPEN SOURCE COMPUTER VISION LIBRARY




                #IMPORT SECTION

# Install a pip package in the current Jupyter kernel
import sys


#importing library computer vision 

###!{sys.executable} -m pip install opencv-python




import cv2

#import os to exit at last


import os


#cmake installation
###!{sys.executable} -m pip install cmake



#import dlib


import dlib
#import face recognition
import face_recognition



#numpy installation

###!{sys.executable} -m pip install numpy
import numpy as np

#for random colors
from random import randrange

#making function for camera face detector
def cam_face():
    #for cam


    #Loading some pre-trained data on face frontals from opencv (haarcascade algo.)
    #its like database of faces you can find many on follow link
    #https://github.com/opencv/opencv/tree/master/data/haarcascades
    data_face = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

    #capturing frames
    cam=cv2.VideoCapture(0)
    #increasing brighness
    cam.set(10,150)
    print("Press Q to Quit Script")

    #to run it forever
    while True:

        #true boolean , frame read
        succesful_frame_read,frame=cam.read()
        #convert to grayscale
        grayscaledcam=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #Face coordinates
        camface_coordinates=data_face.detectMultiScale(grayscaledcam)
        for (x,y,w,h) in camface_coordinates:
            #Draw a rectangle around the face(coordinates,colorBGR,Thickness)
            cv2.rectangle(frame,(x,y),(x+w,y+w),(randrange(150,256),randrange(150,256),(0)),2)






        #show the image of cam

        cv2.imshow("DETFACE",frame)

        #change frames after waiting given argument
        key=cv2.waitKey(1)
        #QTO QUIT
        if key == 81 or key==113:
            break
            print("WRAPED")





    #release the cam
    cam.release()


#cam_face()


#For face recognition 
def facerecognition():
    
    print("SCANNING DATABASE")
    path="FACEIMG"
    images=[]
    names=[]
    imglist=os.listdir(path)
    
    for img in imglist:
        crtimg=cv2.imread(f"{path}/{img}")
        images.append(crtimg)
        names.append(os.path.splitext(img)[0])
    #findencodings
    def encodings(images):
        encodedlist=[]
        for eimg in images:
            cimg=cv2.cvtColor(eimg,cv2.COLOR_BGR2RGB)
            encodings=face_recognition.face_encodings(eimg)[0]
            encodedlist.append(encodings)
        return encodedlist

    scannedfaces=encodings(images)
    
            
    print("SCANNING / ENCODING SUCCESSFUL.!.!.!.!.!.!.!")


    #capture cam frames
    cam=cv2.VideoCapture(0)


    #increasing brighness
    cam.set(10,150)
    
    print("Press Q to Quit Script")

    #to run it forever
    while True:
        success,img=cam.read()
        imgs=cv2.resize(img,(0,0),None,0.25,0.25)
        imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
        CFloc=face_recognition.face_locations(imgs)
        enCF=face_recognition.face_encodings(imgs,CFloc)

        #matching comparing faces
        for encodeface,faceloc in zip (enCF,CFloc):
            matches=face_recognition.compare_faces(scannedfaces,encodeface)
            facedist=face_recognition.face_distance(scannedfaces,encodeface)
            matchindex= np.argmin(facedist)
            
            if matches[matchindex]:
                Name=names[matchindex].upper()
                y1,x2,y2,x1=faceloc
                y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(randrange(150,256),randrange(150,256),0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(randrange(150,256),randrange(150,256),0),cv2.FILLED)
                cv2.putText(img,Name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
                
                
        cv2.imshow("WEBCAM",img)
        #change frames after waiting given argument
        key=cv2.waitKey(1)
        #QTO QUIT
        if key == 81 or key==113:
            break
            print("WRAPED")






    #release the cam
    cam.release()
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    imgmessi=face_recognition.load_image_file("Messi.jpg")
    imgmessi=cv2.cvtColor(imgmessi,cv2.COLOR_BGR2RGB)
    
    faceloc=face_recognition.face_locations(imgmessi)[0]
    faceencoded=face_recognition.face_encodings(imgmessi)[0]
    cv2.rectangle(imgmessi,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),2)
    cv2.putText(imgmessi,"LEONEL MESSI",(10,10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
    
    
    cv2.imshow("MESSI",imgmessi)
    cv2.waitKey(0)
    """
    
#facerecognition()


                        #ASK FOR OPTION
    
    
    

print("[1]DETECT FACES FROM WEBCAM")
print("[2]DETECT AND RECOGNISE FACES FROM WEBCAM"+"(*DATABASE & ENCODING*)")
chosen=input("CHOOSE OPTION 1 or 2 : ")
if int(chosen)==1:
    cam_face()
    
if int(chosen)==2:
    facerecognition()
    
    
