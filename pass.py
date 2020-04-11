# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import dlib
from math import hypot

cv.namedWindow("PASS",cv.WINDOW_NORMAL)

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap=cv.VideoCapture(0)

def midpoint(p1,p2):
    return int((p1.x+p2.x)/2),int((p1.y+p2.y)/2)


def eyes_detect(eye_points,landmarks):
    #horizontal calculation
    lpoint=landmarks.part(eye_points[0]).x,landmarks.part(eye_points[0]).y
    rpoint=landmarks.part(eye_points[3]).x,landmarks.part(eye_points[3]).y
    horizontal=hypot((lpoint[0]-rpoint[0]),(lpoint[1]-rpoint[1]))

    #vertical calculation
    tpoint=midpoint(landmarks.part(eye_points[1]),landmarks.part(eye_points[2]))
    bpoint=midpoint(landmarks.part(eye_points[5]),landmarks.part(eye_points[4]))
    vertical=hypot((tpoint[0]-bpoint[0]),(tpoint[1]-bpoint[1]))
    
    #detecting eyes for display
    eyes=midpoint(landmarks.part(eye_points[0]), landmarks.part(eye_points[3]))
    ex,ey=eyes
    
    #calculating ratios
    ratio=vertical/horizontal
    
    #display on window
    cv.circle(frame,(ex,ey),25,(255,0,0),2)
    cv.line(frame,(tpoint[0],tpoint[1]),(bpoint[0],bpoint[1]),(0,0,255),2)
    cv.line(frame,(lpoint[0],lpoint[1]),(rpoint[0],rpoint[1]),(0,0,255),1)
    
    return ratio

while(cap.isOpened()):
    ret,frame=cap.read()
    
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    faces=detector(gray)
    
    for face in faces:
        x,y=face.left(),face.top()
        x1,y1=face.right(),face.bottom()
        landmarks=predictor(gray,face)
        cv.rectangle(frame,(x,y),(x1,y1),(0,255,0),3)
        right_eye=eyes_detect([36,37,38,39,40,41],landmarks)
        left_eye=eyes_detect([42,43,44,45,46,47],landmarks)
        
        if left_eye<.15 and right_eye<.15:
            cv.putText(frame, "blinked", (10,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),3)
            
            
    cv.imshow("PASS",frame)
    if cv.waitKey(1)==ord("q") & 0xFF:
        break
    
cap.release()
cv.destroyAllWindows()