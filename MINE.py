from cgitb import text
import tkinter
from tkinter import *
import tkinter as tk
from tkinter import font
from turtle import color
from gtts import gTTS
from playsound import playsound
from cgitb import text
from tkinter import *
import cv2
import customtkinter
import cvzone
from PIL import Image, ImageTk, ImageFont, ImageDraw
import keras
from requests import delete
from tensorflow.keras.models import model_from_json
from bidi.algorithm import get_display
import tensorflow
import arabic_reshaper
import mediapipe as mp
import numpy as np
import uuid
import os
import operator
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from keras import layers as ls


customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
 # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("1280x720")
app.title("Signify")
app.configure(background="black")


sign=Tk()
num = 0
# model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

x = base_model.output
x = ls.Flatten()(x)
x = Dense(28, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)


for layer in model.layers[0:20]:
    layer.trainable = False
    

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.load_weights("arsl-2.h5")
# ---------------------------------
sentence = ""

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#image1 = Image.open("ا.gif")
#image11 = ImageTk.PhotoImage(image1)
#image2 = Image.open('pexels-stephan-seeber-1054201(1).jpg')
#image22 = ImageTk.PhotoImage(image2)
#image3 = Image.open('a.jpg')
#image33 = ImageTk.PhotoImage(image3)

cameraon = 0
cap=cv2.VideoCapture(0)
segmentor=SelfiSegmentation(0)
fpsReader=cvzone.FPS()

output_Phrase=""

count_of_writen_letter=0
l=[]
numframe=0
count=0
arabic_imge_width=800
arabic_imge_hight=400
how_many_count_to_write_the_letter=2
how_many_count_frame_to_predict=30
cap = cv2.VideoCapture(0)
cap.set(3,800)
cap.set(4,440)
w=cap.get(3)
h=cap.get(4)

#functions


def show_frames():
    global l, numframe, how_many_count_to_write_the_letter, how_many_count_frame_to_predict, count, w, h, numframe, count_of_writen_letter, output_Phrase, sentence, lcount, cameraon
    if cameraon == 0:
        #camera_label.configure(image = '', width='745',height='492')
        return

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 

        image = cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)



        image = cv2.flip(image, 1)
        # Set flag
        image.flags.writeable = False
        # Detections
        results = hands.process(image)
        # Set flag to true
        image.flags.writeable = True
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        xl=[]
        yl=[]
        zl=[]
        # Rendering results
        if results.multi_hand_landmarks:
            #print("ana hena ya jone")
            for hand in results.multi_hand_landmarks:
                xl=[]
                yl=[]
                zl=[]
                for point in hand.landmark:
                    xl.append(point.x)
                    yl.append(point.y)
                    zl.append(point.z)

            x1=int(min(xl)*w)-80
            y1=int(min(yl)*h)-50
            x2=int(max(xl)*w)+50
            y2=int(max(yl)*h)+50
            if x1<=0:
                x1=0

            if y1<=0:
                y1=0

            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),1)
            prediction1 = dict()
            prediction2 = dict
            if numframe%how_many_count_frame_to_predict==0:
                sign_image=image[y1:y2,x1:x2]
                sign_image=segmentor.removeBG(sign_image,(0,255,0),threshold=.01)
                fpsReader.update(image)
                sign_image=cv2.resize(sign_image,(128,128))    
                #cv2.imshow("test", sign_image)
#                             sign_image = cv2.resize(sign_image, (150, 150))               
                sign_image=sign_image.reshape(1, 128, 128, 3)

                result = model.predict(sign_image)

                #English mapping
                #=========================================================================
                prediction1 = { '7a2'  : result[0][0],
                                    'ain'  : result[0][1],
                                    'alef' : result[0][2],
                                    'ba2'  : result[0][3],
                                    'daal' : result[0][4],
                                    'dad'  : result[0][5],
                                    'faa'  : result[0][6],
                                    'gem'  : result[0][7],
                                    'ghain': result[0][8],
                                    'haa'  : result[0][9],
                                    'kaaf' : result[0][10],
                                    'kha'  : result[0][11],
                                    'laam' : result[0][12],
                                    'meem' : result[0][13],
                                    'noon' : result[0][14],
                                    'qaaf' : result[0][15],
                                    'ra2'  : result[0][16],
                                    'sad'  : result[0][17],
                                    'seen' : result[0][18],
                                    'sheen': result[0][19],
                                    'ta2'  : result[0][20],
                                    'tah'  : result[0][21],
                                    'tha2' : result[0][22],
                                    'waaw' : result[0][23],
                                    'yaa'  : result[0][24],
                                    'zaaa' : result[0][25],
                                    'zain' : result[0][26],
                                    'zal'  : result[0][27]
                    }
                prediction2 = { 'ح'  : result[0][0],
                                    'ع'  : result[0][1],
                                    'ا' : result[0][2],
                                    'ب'  : result[0][3],
                                    'د' : result[0][4],
                                    'ض'  : result[0][5],
                                    'ف'  : result[0][6],
                                    'ج'  : result[0][7],
                                    'غ': result[0][8],
                                    'ه'  : result[0][9],
                                    'ك' : result[0][10],
                                    'خ'  : result[0][11],
                                    'ل' : result[0][12],
                                    'م' : result[0][13],
                                    'ن' : result[0][14],
                                    'ق' : result[0][15],
                                    'ر'  : result[0][16],
                                    'ص'  : result[0][17],
                                    'س' : result[0][18],
                                    'ش': result[0][19],
                                    'ت'  : result[0][20],
                                    'ط'  : result[0][21],
                                    'ث' : result[0][22],
                                    'و' : result[0][23],
                                    'ي'  : result[0][24],
                                    'ظ' : result[0][25],
                                    'ز' : result[0][26],
                                    'ذ'  : result[0][27]
                }
                # Sorting based on top prediction
                #=========================================================================
                prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
                prediction2 = sorted(prediction2.items(), key=operator.itemgetter(1), reverse=True)
                # Sorting based on top prediction
                #append the list

                if(len(l)==0):
                    sentence = sentence + prediction2[0][0]
                    l.append(prediction2[0][0])
                    lcount=count
                if prediction1[0][0]==l[-1]:
                    count=count+1
                if count==how_many_count_to_write_the_letter:
                    output_Phrase=output_Phrase+prediction2[0][0]
                    count_of_writen_letter=count_of_writen_letter+1
                    count=0
                    l=[]
                if count == lcount:
                    count=0
                sentence = sentence + prediction2[0][0]
                l.append(prediction2[0][0])
                #signedTxt.text = sentence
                signedTxt.configure(text = sentence)
                signedTxtWin.configure(text = sentence)
                #en1.delete(0, END)
                #en1.insert(0, sentence)
                
                #print(count)
                print(sentence)
                #series(l)
            #=========================================================================                    
            #show the letter and number of letter that written
            #=========================================================================
            #cv2.putText(image, "{}".format(count_of_writen_letter),(550,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            #cv2.putText(image, prediction1[0][0],(500,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2) 
            fpsReader.update(image)

            #cv2.imshow('Hand Tracking', image)
            #                 print("frame")
            #                 print(numframe)
            numframe=numframe+1
            #show the line of all letter
            #=========================================================================
            
            
            reshaped_text = arabic_reshaper.reshape(output_Phrase)
            #                 reshaped_text = arabic_reshaper.reshape("ãÍãÏ")
            bidi_text = get_display(reshaped_text) 
            #print(reshaped_text)
            
            fontpath = "arial.ttf"
            font = ImageFont.truetype(fontpath, 32)


            img=np.full([arabic_imge_hight,arabic_imge_width,3],0,np.uint8)
            cv2.rectangle(img,(0,0),(arabic_imge_width,arabic_imge_hight),(50,180,30),-1)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((400,180),bidi_text, font = font)
            img = np.array(img_pil)              
            #cv2.imshow('arabic image',img) 
            
            
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    
    
    
    img = Image.fromarray(image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image = img)
    frame.imgtk = imgtk
    frameWin.imgtk = imgtk
    frame.configure(image=imgtk)
    frameWin.configure(image = imgtk)
    
    # Repeat after an interval to capture continiously
    frame.after(20, show_frames)





def start_camera():
    if(buttonStart.text == "Start") :
        buttonStart.configure(text="Pause")
    elif(buttonStart.text == "Pause") :
        buttonStart.configure(text="Start")
    global sentence, cameraon
    if cameraon == 0:
        cameraon = 1
        sentence = ""
        show_frames()
    else:
        play_record()
        cameraon = 0
        frameWin.configure(image = "")
        frame.configure(image = "")
        frame.configure(image = "", width=666, height=387, corner_radius=55)
        frameWin.configure(image = "", width=666, height=387, corner_radius=55)
        #frameWin.configure(image = "")
        
    
def play_record():
 
    language = "ar"
    myobj = gTTS(text = signedTxt.text,
                lang = language,
                slow = False)
    if os.path.exists("audio.mp3"):
        os.remove("audio.mp3")
    myobj.save("audio.mp3")

    playsound('audio.mp3', True)


    
       

def reset(gifpath):
    global file, info, frames, im, anim
    file1 = gifpath
    info = Image.open(file)
    frames = info.n_frames
    im = [tk.PhotoImage(file=file1 ,format=f"gif -index {i}") for i in range(frames - 1)]
    anime = None
    
def series():
#    background_image.lower(gif_label)
    global mylist, slide, file, info, frames, im
    inp = voiceTxt.text
    s = inp.split()
    s[:] = s[::-1]
    s = ' '.join(s)
    inp = s.replace(' ', '')
    mylist = [char for char in inp]
    slide = 0
    global file
    file = str(mylist[0]) + ".gif"
    reset(file)
    idx = 0
    animation(0)


def animation(count):
    
    global anim, slide, mylist
    im2 = im[count]
    circlelbl.configure(image=im2)
    count += 1
    if count >= frames - 1:
        slide += 1
        if slide == len(mylist):
            #circleFrame.lower(background_image)
            return
        #if mylist[slide] == ' ':

        reset(str(mylist[slide]) + ".gif")
        count = 0
       

    anim = app.after(20,lambda :animation(count))


def stop_animation():
    app.after_cancel(anim)
    


def FromVoiceToText():
    import speech_recognition as sr

    r = sr.Recognizer()

    with sr.Microphone() as source:
        
    
        audio = r.listen(source,None,5)
        

    text1 = r.recognize_google(audio, language = 'ar-EG')
    s = text1.split()
    s[:] = s[::-1]
    s = ' '.join(s)
    voiceTxt.configure(text=s)
    # print(text1)
def Delete():
    global sentence
    s = signedTxt.text
    s = s[:-1]
    signedTxt.configure(text = s)
    signedTxtWin.configure(text = s)
    sentence = s
def Clear():
    voiceTxt.configure(text="")
    #en2.delete(0,END)

# def Space():
#     global sentence
#     s = signedTxt.text
#     s +=" "
#     signedTxt.configure(text = s)
#     signedTxtWin.configure(text = s)
#     sentence = s




#design
rect2=customtkinter.CTkFrame(width=35,height=1200,corner_radius=0,fg_color="#D9D9D9")
rect2.place(relx=0.099, rely=0)
rect3=customtkinter.CTkFrame(width=35,height=1280,corner_radius=0,fg_color="#D9D9D9")
rect3.place(relx=0.05, rely=0)
rect4=customtkinter.CTkFrame(width=35,height=1280,corner_radius=0,fg_color="#D9D9D9")
rect4.place(relx=0.894, rely=0)
rect5=customtkinter.CTkFrame(width=35,height=1280,corner_radius=0,fg_color="#D9D9D9")
rect5.place(relx=0.946, rely=0)

#buttons
rect1=customtkinter.CTkFrame(width=2000,height=85,corner_radius=0,fg_color="#D9D9D9",bg_color="#D9D9D9")
rect1.place(relx=0, rely=0.815)

buttonClr = customtkinter.CTkButton(text="Clear",width=190,height=56,corner_radius=23,bg_color="#D9D9D9", command = Clear)
buttonClr.place(relx=0.190625, rely=0.83472)

buttonRec = customtkinter.CTkButton(text="Hold to record",width=190,height=56,fg_color="#A83B3B",corner_radius=23,bg_color="#D9D9D9",command=FromVoiceToText)
buttonRec.place(relx=0.42578125, rely=0.83472)

buttonSnd = customtkinter.CTkButton(text="send",width=190,height=56,corner_radius=23,bg_color="#D9D9D9", command=series)
buttonSnd.place(relx=0.6609375, rely=0.83472)

#camera frame "model"
frame = customtkinter.CTkLabel(width=666,
                            height=387,
                            corner_radius=55,
                            fg_color='#D9D9D9')
frame.place(relx=0.2453125, rely=0.095833)


signedTxt = customtkinter.CTkLabel(width=666,
                               height=43,
                               fg_color=('black'),
                               text=""
                               )
signedTxt.place(relx=0.2453125, rely=0.47)

#voice text
voiceTxt = customtkinter.CTkLabel(text="",
                                 width=400,
                                 height=48,
                                 fg_color=('black'),#D9D9D9
                                 corner_radius=10,
                                 bg='#fff')
voiceTxt.place(relx=0.34375, rely=0.7)


#second window
window = Toplevel(app)
window.geometry("1600x907")
window.configure(background='#000000')


# buttons
rect6=customtkinter.CTkFrame(window,width=2000,height=85,corner_radius=0,fg_color="#D9D9D9",bg_color="#D9D9D9")
rect6.place(relx=0, rely=0.815)

buttonDel = customtkinter.CTkButton(window,text="Delete",width=190,height=56,corner_radius=23,bg_color="#D9D9D9",command=Delete)
buttonDel.place(relx=0.190625, rely=0.83472)

buttonStart = customtkinter.CTkButton(window,text="Start",width=190,height=56,corner_radius=23,bg_color="#D9D9D9",command=start_camera)
buttonStart.place(relx=0.42578125, rely=0.83472)

buttonSpc = customtkinter.CTkButton(window,text="Space",width=190,height=56,corner_radius=23,bg_color="#D9D9D9")
buttonSpc.place(relx=0.6609375, rely=0.83472)


#camera frame "model"
frameWin = customtkinter.CTkLabel(window,width=666,
                            height=387,
                            corner_radius=55,
                            fg_color='#D9D9D9')
frameWin.place(relx=0.43359375, rely=0.1361111)


signedTxtWin = customtkinter.CTkLabel(window,
                               width=666,
                               height=43,
                               fg_color=('black'),
                               text=""
                               )
signedTxtWin.place(relx=0.43359375, rely=0.51)


#animation

circlelbl = customtkinter.CTkLabel(window,width=410,height=387,corner_radius=50,fg_color="#D9D9D9")
circlelbl.place(relx=0.07734375, rely=0.136111)








app.mainloop()