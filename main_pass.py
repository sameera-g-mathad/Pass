# -*- coding: utf-8 -*-
import dlib
import cv2 as cv
from math import hypot
import numpy as np
import textwrap
import enchant
import requests
import pyttsx3
import os
import random
from pygame import mixer
from mutagen.id3 import ID3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

cv.namedWindow("PASS",cv.WINDOW_NORMAL)
cv.resizeWindow("PASS",10000,10000)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img=np.zeros((500,500,3),np.uint8)
img.fill(255)
keyboard=np.zeros((600,1000,3),np.uint8)
engine=pyttsx3.init()
engine.setProperty("rate", 150)
rgaze_sec=0
lgaze_sec=0
seconds=0
numbers=["0","1","2","3","4","5","6","7","8","9"]
alpa_set1=["Q","W","E","R","T","Y","U","I","O","P"]
alpa_set2=["A","S","D","F","G","H","J","K","L","SP"]
alpa_set3=["Z","X","C","V","B","N","M",",",".","DL"]
utils=["CLEAR","EMAIL","EXIT"]
suggested=["","","","","","","","","",""]
keys=[suggested,numbers,alpa_set1,alpa_set2,alpa_set3,utils]
selected=[]
row=2
active=0
msg=""
text=""
url="http://127.0.0.1:8000"
mailoptions={"toEmail":"pruthvilakkur3@gmail.com"}
exiting=False
suggestions=enchant.Dict("en_US")




model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


model.load_weights('model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

anime_index=0
animations=[]
detect_index=0
song_index=0
songs=[]
song_name=""
song_exit=False
directory="C:/Users/Asus/Pass/"


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
    width=333
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
            width_inc+=333
    return

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
    #cv.circle(frame,(ex,ey),15,(255,0,0),2)
    # cv.line(frame,(tpoint[0],tpoint[1]),(bpoint[0],bpoint[1]),(0,0,255),2)
    # cv.line(frame,(lpoint[0],lpoint[1]),(rpoint[0],rpoint[1]),(0,0,255),1)
    
    return ratio
def eye_gaze(eyepoints,landmark,frame):
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
def music_system(emotion):
    global songs,anime_index,detect_index,song_index,songs,directory,song_exit
    directory=directory+emotion
    os.chdir(directory)
    for file in os.listdir(directory):
        if file.endswith(".mp3"):
            songs.append(file)
        if file.endswith(".mp4"):
            animations.append(file)
  
    blink_sec=0
    rightgaze_sec=0
    leftgaze_sec=0
    
    cap2=[cv.VideoCapture(i) for i in animations]
    cap=cv.VideoCapture(0)
    play_song()
    while cap.isOpened():
        cv.resizeWindow("PASS",5000,5000)
        ret2,video=cap2[anime_index].read()
        ret,frame=cap.read()
        #cv.imshow("video",frame)
        if ret2==True:
            cv.putText(video,song_name,(10,50), cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),2)
            cv.imshow("PASS",video)
            if (cv.waitKey(1)==ord('q') & 0xFF) or song_exit:
                 break
        else:
            anime_index=random.randint(0,(len(animations)-1))
            detect_index+=1
            if detect_index>=len(animations):
                cap2=[cv.VideoCapture(i) for i in animations]
                detect_index=0
        
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        faces=detector(gray)
        
        for face in faces:
            landmarks=predictor(gray,face)
            
            right_eye=eyes_detect([36,37,38,39,40,41],landmarks)
            left_eye=eyes_detect([42,43,44,45,46,47],landmarks)
            if left_eye<.16 and right_eye<.16:
                blink_sec+=1
                if blink_sec==20:
                    print("releasing")
                    mixer.music.stop()
                    cap.release()
                    cap2[anime_index].release()
                    song_exit=True
                    break
            
            else:
                blink_sec=0
            right_eye_gaze=eye_gaze([36,37,38,39,40,41],landmarks,frame)
            left_eye_gaze=eye_gaze([42,43,44,45,46,47],landmarks,frame)
            
            avg_gaze=(left_eye_gaze+right_eye_gaze)/2
            
            if avg_gaze<.5:
                rightgaze_sec+=1
                if rightgaze_sec==20:
                    print("right")
                    song_index+=1
                    if song_index==len(songs):
                        song_index=0
                    play_song()
            else:
                rightgaze_sec=0
            if avg_gaze>1.5:
                leftgaze_sec+=1
                if leftgaze_sec==20:
                    print("left")
                    if song_index==0:
                        song_index=len(songs)-1
                    else:
                        song_index-=1
                    play_song()
            else:
                leftgaze_sec=0
    mixer.music.stop()
    directory="C:/Users/Asus/Pass/"
    song_exit=False
    #cv.destroyAllWindows()
    
    
    
    
def play_song():
    global song_name
    reldir=os.path.realpath(songs[song_index])
    audio=ID3(reldir)
    song_name=audio["TIT2"].text[0]
    mixer.init()
    mixer.music.load(songs[song_index])
    mixer.music.play()
    
    
    
def emotion_detection():
    cv.resizeWindow("PASS",5000,5000)
    cap = cv.VideoCapture(0)
    frames=0
    compare=""
    
    while True:
        # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            cv.putText(frame, "How Are You Feeling????",(125,50),cv.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
            
            #facecasc = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces=detector(gray)
            #faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
            
            for face in faces:
                cv.rectangle(frame, (face.left(), face.top()+5), (face.right(), face.bottom()), (0, 255, 0), 2)
                frame=cv.resize(frame,(500,1000))       
                cv.imshow('PASS', frame)
                roi_gray = gray[face.top():face.bottom(), face.left():face.right()]
                cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                if(compare==emotion_dict[maxindex]):
                    frames+=1
                else:
                    compare=emotion_dict[maxindex]
                    frames=0
                if(frames==20):
                    print(emotion_dict[maxindex])      
                    engine.say("mood detected "+emotion_dict[maxindex])
                    engine.runAndWait()
                    cv.putText(frame, emotion_dict[maxindex],(face.left(), face.top()), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                    if(emotion_dict[maxindex]=="Happy" or emotion_dict[maxindex]=="Surprised" ):
                        cap.release()
                        music_system("positive_emotions")
                    if(emotion_dict[maxindex]=="Sad" or emotion_dict[maxindex]=="Angry" or emotion_dict[maxindex]=="Disgusted" or emotion_dict[maxindex]=="Fearful" ):
                        cap.release()
                        music_system("negative_emotions")
                    if(emotion_dict[maxindex]=="Neutral"):
                        choose=random.randint(1, 2)
                        cap.release()
                        if(choose==1):
                            music_system("positive_emotions")
                        if(choose==2):
                            music_system("negative_emotions")
                    engine.say("exiting the sentiment based music player")
                    engine.runAndWait()
                    return
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    #cv.destroyAllWindows()
    
    return
def blink_to_text():
    global text,seconds,row,active,msg,rgaze_sec,lgaze_sec,engine,selected,exiting,url,mailoptions,img,suggestions,suggested
    cap=cv.VideoCapture(0)
    keyboardf()
    while(cap.isOpened()):
        cv.resizeWindow("PASS",10000,10000)
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
            
            if left_eye<.16 and right_eye<.16:
                seconds+=1
                if(seconds==3):
                    #cv.putText(frame, "blinked", (100,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),3)
                    if row==0:
                            msg_split=msg.split(" ")
                            msg_split[-1]=selected[active]
                            msg=" "
                            msg=msg.join(msg_split)
                            img.fill(255)
                            engine.say(selected[active])
                    else:
                            if selected[active]=="":
                                pass
                            if selected[active]=="SP":
                                msg+=" "
                                engine.say("space")
                               
                            elif selected[active]=="CLEAR":
                                msg=""
                                suggested=["","","","","","","","","",""]
                                engine.say("cleared")
                                
                            elif selected[active]=="DL":
                                delchar=list(msg)
                                del[delchar[-1]]
                                msg="".join(delchar)
                                img.fill(255)
                                engine.say("delete")
                               
                            elif(row==5 and selected[active]=="EXIT"):
                                exiting=True
                                
                            elif (row==5 and selected[active]=="EMAIL"):
                                try:
                                    if(msg!=""):
                                        mailoptions["message"]=msg
                                        response=requests.post(url,params=mailoptions)
                                        print(response.status_code)
                                        msg=""
                                        img.fill(255)
                                        engine.say("email sent successfully")
                                except:
                                    engine.say("something went wrong while sending the email") 
                                    
                            else:
                                msg+=selected[active] 
                                message=msg.split(" ")
                                suggestions.check(message[-1])
                                words=suggestions.suggest(message[-1])
                                words=words[0:10]
                                suggested[0:len(words)]= words
                                keyboardf()
                                engine.say(selected[active])
                    
                    text=textwrap.wrap(msg,width=25)
            else:
                seconds=0
                
                
                
            right_eye_gaze=eye_gaze([36,37,38,39,40,41],landmarks,frame)
            left_eye_gaze=eye_gaze([42,43,44,45,46,47],landmarks,frame)
            
            avg_gaze=(left_eye_gaze+right_eye_gaze)/2
           
            if(avg_gaze<.5):
                #cv.putText(frame, "right", (10,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,255),3)
                rgaze_sec+=1
                if rgaze_sec==10:
                    active+=1
                    if active>9:
                        active=0
                        row+=1
                        if row>5:
                            row=0
                        selected=keys[row]
                    if row==5 and active>2:
                        active=0
                        row=0
                        selected=keys[row]
                    keyboardf()
                    rgaze_sec=0
                
            elif(avg_gaze>.5 and avg_gaze<1.5):
               # cv.putText(frame, "center", (10,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,2,255),3)
                selected=keys[row]
            else:
                #cv.putText(frame, "left", (10,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,5,255),3)
                lgaze_sec+=1
                if lgaze_sec==10:
                    active-=1
                    if active<0:
                        active=9
                        row-=1
                        if row<0:
                            row=5
                        if row==5:
                            active=2
                    selected=keys[row]
                   
                    keyboardf()
                    lgaze_sec=0
                #print("left",avg_gaze)
        textx=0
        texty=50
        for i in text:
            cv.putText(img,i,(textx,texty),cv.FONT_HERSHEY_TRIPLEX,1,(84,83,81),2)
            texty+=50
        engine.runAndWait()
        upper_window=np.concatenate((img,frame),axis=1)
        window=np.concatenate((upper_window,keyboard),axis=0)
        cv.imshow("PASS",window)
        if (cv.waitKey(1)==ord("q") & 0xFF) or exiting:
            break
        
    cap.release()
    #cv.destroyAllWindows()
    engine.say("exiting the blink to text communication")
    engine.runAndWait()
    return

def main_pass():
    cap=cv.VideoCapture(0)
    blink_sec=0
    rightgaze_sec=0
    leftgaze_sec=0
    main_exit=False
    while (cap.isOpened()):
        cv.resizeWindow("PASS",10000,10000)
        ret,frame=cap.read()
        cv.putText(frame, "Pass Application", (200,50), cv.FONT_HERSHEY_DUPLEX, 1, (0,255,0),2)
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces=detector(gray)
        for face in faces:
            landmarks=predictor(gray,face)
            cv.putText(frame, "Sentiment-Based", (10,200), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255),2)
            cv.putText(frame, "Music System", (10,230), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255),2)
            cv.putText(frame, "Blink To Text", (430,200), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),2)
            cv.putText(frame, "Communication", (430,230), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),2)
            
            right_eye=eyes_detect([36,37,38,39,40,41],landmarks)
            left_eye=eyes_detect([42,43,44,45,46,47],landmarks)
            if left_eye<.16 and right_eye<.16:
                blink_sec+=1
                if blink_sec==20:
                    print("releasing")
                    cap.release()
                    main_exit=True
                    break
            
            else:
                blink_sec=0
            right_eye_gaze=eye_gaze([36,37,38,39,40,41],landmarks,frame)
            left_eye_gaze=eye_gaze([42,43,44,45,46,47],landmarks,frame)
            
            avg_gaze=(left_eye_gaze+right_eye_gaze)/2
            
            if avg_gaze<.5:
                rightgaze_sec+=1
                if rightgaze_sec==20:
                    print("right")
                    cap.release()
                    engine.say("starting blink to text communication")
                    engine.runAndWait()
                    blink_to_text()
                    cap=cv.VideoCapture(0)
                    
            else:
                rightgaze_sec=0
            if avg_gaze>1.5:
                leftgaze_sec+=1
                if leftgaze_sec==20:
                    print("left")   
                    cap.release()
                    engine.say("starting sentiment based music player")
                    engine.runAndWait()
                    emotion_detection()
                    cap=cv.VideoCapture(0)
            else:
                leftgaze_sec=0
        frame=cv.resize(frame,(500,1000))        
        cv.imshow("PASS",frame)
        if cv.waitKey(1) & 0xFF == ord('q') or main_exit:
                break
    cap.release()
    cv.destroyAllWindows()
    engine.say("quitting the pass application")
    engine.runAndWait()

main_pass()