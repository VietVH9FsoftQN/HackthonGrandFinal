import eventlet
import json
from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import io
import os
from PIL import Image
#######################################
from chatbot import chatbot
from chatbot.chatbot import response,classify
#######################################
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from random import randrange
from facedetect_folder.facedetect import detect_face,reset_keras
from keras.models import load_model
from statistics import mode
from facedetect_folder.utils.datasets import get_labels
from facedetect_folder.utils.inference import detect_faces
from facedetect_folder.utils.inference import draw_text
from facedetect_folder.utils.inference import draw_bounding_box
from facedetect_folder.utils.inference import apply_offsets
from facedetect_folder.utils.inference import load_detection_model
from facedetect_folder.utils.preprocessor import preprocess_input
import face_recognition
import code_tfpose
from code_tfpose import *
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import gc
import pyttsx3
from gtts import gTTS
import os
import record_chatbot
from record_chatbot import *
#from text_to_speech import text_2_speech
#######################################
# eventlet.monkey_patch()
app = Flask(__name__)
socketio = SocketIO(app)
@app.route('/')
def index():        # set index
    return render_template('index.html')
##############################################ACTIVE DETECT FACE##################################################
@socketio.on('active_face2')
def activez_face(json_str):
    data = json_str
    if data=='active_face':
        global emotion_model_path 
        emotion_model_path = 'facedetect_folder/models/emotion_model.hdf5'
        global emotion_labels 
        emotion_labels = get_labels('fer2013')
        global emotion_classifier
        emotion_classifier = load_model(emotion_model_path)
        global kn_image
        global kn_face_encoding_0
        global known_face_encodings
        known_face_encodings=[]
        global known_face_names_0
        global known_face_names
        known_face_names=[]
        
        files = [f for f in os.listdir("facedetect_folder/faceknow")]
        for i in files:
            kn_image=face_recognition.load_image_file("facedetect_folder/faceknow/"+i)
            kn_face_encoding_0 = face_recognition.face_encodings(kn_image)[0]
            known_face_encodings.append(kn_face_encoding_0)
            known_face_names_0=i[:-4]
            known_face_names.append(known_face_names_0)
        
        print(8*'active_face')
        repliesmess="done_active"
        socketio.emit('done_active_face', data=repliesmess)
##############################################ACTIVE DETECT FACE##################################################
@socketio.on('publish')     # send mess
def handle_publish(json_str):
    data = json_str
    ##################### (nhớ mở những dòng ở giữa đây)
    image_data = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
    #####################
    #####################
    file_send,name,emotion_text=detect_face(image_data,known_face_names,known_face_encodings,emotion_classifier,emotion_labels)
    #####################
    #####################
    socketio.emit('mqtt_message', data=file_send)
    # #####################
    if (name != 'Unknown'): 
        check_time=time.time()

        socketio.emit('check_time', data=check_time)
        socketio.emit('mqtt_message_name', data=name)
        socketio.emit('mqtt_message_emotion', data=emotion_text)
        # text="Hello "+name+", I see your mood today " + emotion_text+". Would you like to talk with me?"

        # text_2_speech(text)
#         socketio.emit('send_audio_file', data="../static/audio/ouput_speech.wav")
    # #####################
##############################################DEACTIVE DETECT FACE##################################################

@socketio.on('deactive_face')
def deactive_face(json_str):
    data = json_str
    if data=='deactive_face':
        ######################
        sess = get_session()
        clear_session()
        sess.close()
        sess = get_session()

        try:
            del emotion_model_path
            del emotion_labels
            del emotion_classifier
            del kn_image
            del kn_face_encoding
            del known_face_encodings
            del known_face_names
        except:
            pass

        print(gc.collect()) # if it's done something you should see a number being outputted

        # use the same config as you used to create the session
        config = tensorflow.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        set_session(tensorflow.Session(config=config))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        print(60*'@')
        
        ######################
        repliesmess="done_deactive"
        print(repliesmess)
        socketio.emit('done_deactive_face', data=repliesmess)


##############################################DEACTIVE DETECT FACE##################################################


##############################################ACTIVE TRAIN##################################################
@socketio.on('active_train')
def active_train_zz(json_str):
        data = json_str
        global action_labels
        global my_detector
        global multipeople_classifier,multiperson_tracker
        # if data=='active_train1':

        # ######################
                
        # action_labels =  ['jump','kick','punch','run','sit','squat','stand','walk','none']
        action_labels =  ['none','none','none','none','sit','squat','stand','none','none']
        
        my_detector = SkeletonDetector(OpenPose_MODEL, image_size)
        
        multipeople_classifier,multiperson_tracker=multi(LOAD_MODEL_PATH,action_labels)
        print(80*'*')
        ######################
        repliesmess="done_active"
        print(repliesmess)
        print(80*'*')     
        socketio.emit('done_active_train', data='done_active')
        socketio.emit('done_select_td', data='4')

@socketio.on('publish_train')     # send mess
def handle_publish_train(json_str):
    data = json_str
    ######################
    image_data = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
    ######################
    #*******************************************************************************************************#
    #*******************************************************************************************************#
    ######################
    count_human,label,file_send=code_main(my_detector,multipeople_classifier,multiperson_tracker,image_data)
    ######################
    #*******************************************************************************************************#
    ######################
    socketio.emit('mqtt_message', data=file_send)
    print(2*'count_human',count_human)
    if count_human!=0:
        if (label!='none') and (label!='sit'):
                socketio.emit('label_human', data=label)
        check_time=time.time()
        socketio.emit('check_time_train', data=check_time)
    if count_human=='none':
    #     time.sleep(20)
        check_time=time.time()
        socketio.emit('check_time_canhbao', data=check_time)
        print('GUI CANH BAO KHONG THAY NGUOI GUI CANH BAO')
        socketio.emit('send_audio_file', data="../static/audio/warning.wav")

##############################################ACTIVE TRAIN##################################################
##############################################DEACTIVE DETECT TRAIN##################################################
@socketio.on('de_active_train')
def deactive_train(json_str):
    data = json_str
    if data=='deactive_train':
        ######################
        sess = get_session()
        clear_session()
        sess.close()
        sess = get_session()
        try:
            del my_detector
            del multipeople_classifier
            del multiperson_tracker       
        except:
            pass
        print(gc.collect()) # if it's done something you should see a number being outputted

        # use the same config as you used to create the session
        config = tensorflow.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        set_session(tensorflow.Session(config=config))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        ######################
        print(60*'k')
        repliesmess="done_deactive"
        print(repliesmess)
        socketio.emit('done_deactive_train', data=repliesmess)
##############################################DEACTIVE DETECT TRAIN##################################################
@socketio.on('check_sendmess_spe')
def handle_sendmess(json_str):
    repliesmess=""
    data = json_str
    print("+++++++++")
    print(data,type(data))
    # repliesmess=response(data)
    if (data=='blob_blob'):
        socketio.emit('replies_sendmess_spe', data='replies2_sendmess_spe')
# ######################
#########them chat bot tieng anh vao day####################
@socketio.on('sendmess')
def handle_sendmess(json_str):
    repliesmess=""
    data = json_str
    print("+++++++++")
    print(data,type(data))
    # repliesmess=response(data)
    repliesmess="Yes, I have"
    socketio.emit('replieszzz', data=repliesmess)
###################### 

@socketio.on('sendmess_spe')
def handle_sendmess_spe(json_str):
    repliesmess=""
    data = json_str
   # print(100*'*')
   # print(data,type(data))
    try:
        os.remove('myfile.wav')
    except:
        print('removed')
    with open('myfile.wav', mode='bx') as f:
        f.write(data)
    
    name=speech_to_speech('myfile.wav')
    print(name)
    socketio.emit('send_process_speech', data="../static/audio/"+name)    
#     socketio.emit('replieszzz', data='Get audio speech')
#    socketio.emit('process_speech', data='Get_audio_speech')
    
######################  

@socketio.on('check_point')
def process_spe(json_str):
    text='You make' + json_str + "movements.Your score is"+json_str
    name=text_to_speech_1(text)
    socketio.emit('send_process_speech', data="../static/audio/"+name)

if __name__ == '__main__':
    socketio.run(app,host='0.0.0.0', port=5000,debug=True, keyfile='key.pem', certfile='cert.pem',use_reloader=False)

