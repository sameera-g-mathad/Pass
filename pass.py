# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import dlib
from math import hypot
import textwrap
import enchant
import time
cv.namedWindow("PASS",cv.WINDOW_NORMAL)
cv.resizeWindow("PASS",10000,10000)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img=np.zeros((500,500,3),np.uint8)
img.fill(255)
keyboard=np.zeros((600,1000,3),np.uint8)
frames=0
seconds=0
numbers=["0","1","2","3","4","5","6","7","8","9"]
alpa_set1=["Q","W","E","R","T","Y","U","I","O","P"]
alpa_set2=["A","S","D","F","G","H","J","K","L","SP"]
alpa_set3=["Z","X","C","V","B","N","M",",",".","DL"]
utils=["SEND MAIL","CLEAR"]
suggested=["","","","","","","","","",""]
keys=[suggested,numbers,alpa_set1,alpa_set2,alpa_set3,utils]
selected=[]
row=2
active=0
msg=""
text=""
suggestions=enchant.Dict("en_US")
    
def keyboardf():
    keyboard[:]=(104,110,103)
    width=100
    height=100
    x=0
    y=0
    width_inc=0
    for i in range(len(suggested)):
        if active==i and row==0:
            cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(255,255,255),-1)
        else:
            cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(204,236,97),2)
        width_x,height_y=cv.getTextSize(suggested[i],cv.FONT_HERSHEY_SIMPLEX,2,2)[0]
        center_x=(x+width_inc)+5
        center_y=int((height)/2)
        cv.putText(keyboard,suggested[i],(center_x,center_y),cv.FONT_HERSHEY_PLAIN,1.1,(204,236,97),1,cv.LINE_AA)
        width_inc+=100
    width=100
    height=100
    x=0
    y=100
    width_inc=0
    for i in range(len(numbers)):
        if active==i and row==1:
            cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(255,255,255),-1)
        else:
            cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(104,187,95),2)
        width_x,height_y=cv.getTextSize(numbers[i],cv.FONT_HERSHEY_SIMPLEX,2,2)[0]
        center_x=int((width-width_x)/2)+(x+width_inc)
        center_y=int((height+height_y)/2)+y
        cv.putText(keyboard,numbers[i],(center_x,center_y),cv.FONT_HERSHEY_COMPLEX,2,(104,187,95),2,cv.LINE_AA)
        width_inc+=100
    width=100
    height=100
    x=0
    y=200
    width_inc=0
    for i in range(len(alpa_set1)):
           if active==i and row==2:
               cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(255,255,255),-1)
           else:
               cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(75,105,242),2)
              
           width_x,height_y=cv.getTextSize(alpa_set1[i],cv.FONT_HERSHEY_SIMPLEX,2,2)[0]
           center_x=int((width-width_x)/2)+(x+width_inc)
           center_y=int((height+height_y)/2)+y
           cv.putText(keyboard,alpa_set1[i],(center_x,center_y),cv.FONT_HERSHEY_COMPLEX,2,(75,105,242),2,cv.LINE_AA)
           width_inc+=100
    width=100
    height=100
    x=0
    y=300
    width_inc=0
    for i in range(len(alpa_set2)):
           if active==i and row==3:
               cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(255,255,255),-1)
           else:
               cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(240,174,74),2)
              
           width_x,height_y=cv.getTextSize(alpa_set2[i],cv.FONT_HERSHEY_SIMPLEX,2,2)[0]
           center_x=int((width-width_x)/2)+(x+width_inc)
           center_y=int((height+height_y)/2)+y
           cv.putText(keyboard,alpa_set2[i],(center_x,center_y),cv.FONT_HERSHEY_COMPLEX,2,(240,174,74),2,cv.LINE_AA)
           width_inc+=100
    width=100
    height=100
    x=0
    y=400
    width_inc=0
    for i in range(len(alpa_set3)):
           if active==i and row==4:
              cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(255,255,255),-1)
           else:
               cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(74,214,240),2)
              
           width_x,height_y=cv.getTextSize(alpa_set3[i],cv.FONT_HERSHEY_SIMPLEX,2,2)[0]
           center_x=int((width-width_x)/2)+(x+width_inc)
           center_y=int((height+height_y)/2)+y
           cv.putText(keyboard,alpa_set3[i],(center_x,center_y),cv.FONT_HERSHEY_COMPLEX,2,(74,214,240),2,cv.LINE_AA)
           width_inc+=100
    width=500
    height=100
    x=0
    y=500
    width_inc=0
    for i in range(len(utils)):
            if active==i and row==5:
                cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(255,255,255),-1)
            else:
                cv.rectangle(keyboard,(x+width_inc,y),(width+width_inc,height+y),(99,82,241),2)
              
            width_x,height_y=cv.getTextSize(utils[i],cv.FONT_HERSHEY_SIMPLEX,2,2)[0]
            center_x=int((width-width_x)/2)+(x+width_inc)
            center_y=int((height+height_y)/2)+y
            cv.putText(keyboard,utils[i],(center_x,center_y),cv.FONT_HERSHEY_COMPLEX,2,(99,82,241),2,cv.LINE_AA)
            width_inc+=500
    return
    
cap=cv.VideoCapture(0)
keyboardf()
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
    # cv.line(frame,(tpoint[0],tpoint[1]),(bpoint[0],bpoint[1]),(0,0,255),2)
    # cv.line(frame,(lpoint[0],lpoint[1]),(rpoint[0],rpoint[1]),(0,0,255),1)
    
    return ratio
def eye_gaze(eyepoints,landmark):
    eye_region=np.array([
            (landmark.part(eyepoints[0]).x,landmark.part(eyepoints[0]).y),
            (landmark.part(eyepoints[1]).x,landmark.part(eyepoints[1]).y),
            (landmark.part(eyepoints[2]).x,landmark.part(eyepoints[2]).y),
            (landmark.part(eyepoints[3]).x,landmark.part(eyepoints[3]).y),
            (landmark.part(eyepoints[4]).x,landmark.part(eyepoints[4]).y),
            (landmark.part(eyepoints[5]).x,landmark.part(eyepoints[5]).y),
        ]) 
    min_x=np.min(eye_region[:,0])
    max_x=np.max(eye_region[:,0])
    min_y=np.min(eye_region[:,1])
    max_y=np.max(eye_region[:,1])
    # print(eye_region)
    # print(min_x,max_x,min_y,max_y)
    
    eyes=frame[min_y:max_y,min_x:max_x]
    
    gray_eye=cv.cvtColor(eyes,cv.COLOR_BGR2GRAY)
    ret,threshold=cv.threshold(gray_eye,50,255,cv.THRESH_BINARY)
    

    eye_height,eye_width=threshold.shape
    
    left_gaze=threshold[0:eye_height,0:int(eye_width/2)]
    right_gaze=threshold[0:eye_height,int(eye_width/2):eye_width]
    # top_gaze=threshold[0:int(eye_height/2),0:eye_width]
    # bottom_gaze=threshold[int(eye_height/2):eye_height,0:eye_width]
    
    left_white=cv.countNonZero(left_gaze)
    right_white=cv.countNonZero(right_gaze)
    # top_white=cv.countNonZero(top_gaze)
    # bottom_white=cv.countNonZero(bottom_gaze)
    
    if left_white!=0 and right_white!=0:
        gaze_ratio=left_white/right_white
        return gaze_ratio
    return 1
while(cap.isOpened()):
    ret,frame=cap.read()
    if len(text)==0:
        img.fill(255)
    frame=cv.resize(frame,(500,500))
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    faces=detector(gray)
    
    for face in faces:
        x,y=face.left(),face.top()
        x1,y1=face.right(),face.bottom()
        cv.rectangle(frame,(x,y),(x1,y1),(0,255,0),3) 
        
        landmarks=predictor(gray,face)
        
        right_eye=eyes_detect([36,37,38,39,40,41],landmarks)
        left_eye=eyes_detect([42,43,44,45,46,47],landmarks)
        #print(left_eye,right_eye)
        if left_eye<.15 and right_eye<.15:
            time.sleep(1)
            cv.putText(frame, "blinked", (100,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),3)
            frames-=2
            if row==0:
                    msg_split=msg.split(" ")
                    msg_split[-1]=selected[active]
                    msg=" "
                    msg=msg.join(msg_split)
                    img.fill(255)
            else:
                    if selected[active]=="":
                        pass
                    if selected[active]=="SP":
                        msg+=" "
                    elif selected[active]=="CLEAR":
                        msg=""
                    # if selected[active]=="DL":
                    #     message=msg.split(" ")
                    #     delword=message[-1]
                        
                    else:
                        msg+=selected[active] 
                        message=msg.split(" ")
                        suggestions.check(message[-1])
                        words=suggestions.suggest(message[-1])
                        words=words[0:10]
                        suggested[0:len(words)]= words
                        keyboardf()
            text=textwrap.wrap(msg,width=25)
               
            
            
            
        right_eye_gaze=eye_gaze([36,37,38,39,40,41],landmarks)
        left_eye_gaze=eye_gaze([42,43,44,45,46,47],landmarks)
        
        avg_gaze=(left_eye_gaze+right_eye_gaze)/2
       
        if(avg_gaze<.5):
            #cv.putText(frame, "right", (10,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,255),3)
            frames+=1
            if frames==10:
                active+=1
                if active>9:
                    active=0
                    row+=1
                    if row>5:
                        row=0
                    selected=keys[row]
                if row==5 and active>1:
                    active=0
                    row=0
                    selected=keys[row]
                keyboardf()
                frames=0
            
        elif(avg_gaze>.5 and avg_gaze<2):
           # cv.putText(frame, "center", (10,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,2,255),3)
            selected=keys[row]
        else:
            #cv.putText(frame, "left", (10,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,5,255),3)
            frames+=1
            if frames==10:
                active-=1
                if active<0:
                    active=9
                    row-=1
                    if row<0:
                        row=5
                    if row==5:
                        active=1
                selected=keys[row]
               
                keyboardf()
                frames=0
            #print("left",avg_gaze)
    textx=0
    texty=50
    for i in text:
        cv.putText(img,i,(textx,texty),cv.FONT_HERSHEY_TRIPLEX,1,(84,83,81),2)
        texty+=50
    upper_window=np.concatenate((img,frame),axis=1)
    window=np.concatenate((upper_window,keyboard),axis=0)
    cv.imshow("PASS",window)
    if cv.waitKey(1)==ord("q") & 0xFF:
        break
    
cap.release()
cv.destroyAllWindows()