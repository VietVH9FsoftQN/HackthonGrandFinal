import eventlet
import os
import json
from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import io
from OpenSSL import SSL, crypto
from PIL import Image
#######################################
# from chatbot import chatbot
# from chatbot.chatbot import response,classify
#######################################
from cv2 import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from random import randrange

#######################################
# from facedetect_folder.facedetect import detect_face,reset_keras
# from keras.models import load_model
# from statistics import mode
# from facedetect_folder.utils.datasets import get_labels
# from facedetect_folder.utils.inference import detect_faces
# from facedetect_folder.utils.inference import draw_text
# from facedetect_folder.utils.inference import draw_bounding_box
# from facedetect_folder.utils.inference import apply_offsets
# from facedetect_folder.utils.inference import load_detection_model
# from facedetect_folder.utils.preprocessor import preprocess_input
# import face_recognition

# import code_tfpose
# from code_tfpose import *



# from keras.backend.tensorflow_backend import set_session
# from keras.backend.tensorflow_backend import clear_session
# from keras.backend.tensorflow_backend import get_session
# import tensorflow
# import gc

#######################################

eventlet.monkey_patch()
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
        #     #sua lai phan active face known_face_encodings,known_face_names

        # global emotion_model_path 
        # emotion_model_path = 'facedetect_folder/models/emotion_model.hdf5'
        # global emotion_labels 
        # emotion_labels = get_labels('fer2013')
        # global emotion_classifier
        # emotion_classifier = load_model(emotion_model_path)
        # global kn_image
        # kn_image = face_recognition.load_image_file("facedetect_folder/faceknow/test.jpg")
        # global kn_face_encoding
        # kn_face_encoding = face_recognition.face_encodings(kn_image)[0]
        # global known_face_encodings
        # known_face_encodings = [kn_face_encoding]
        # global known_face_names
        # known_face_names = ["An"]
        
        print(8*'active_face')
        repliesmess="done_active"
        socketio.emit('done_active_face', data=repliesmess)
##############################################ACTIVE DETECT FACE##################################################
@socketio.on('publish')     # send mess
def handle_publish(json_str):
    data = json_str
    # ##################### (nhớ mở những dòng ở giữa đây)
    # image_data = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
    # #####################
    # #####################
    # file_send,name,emotion_text=detect_face(image_data,known_face_names,known_face_encodings,emotion_classifier,emotion_labels)
    # #####################
    # #####################
    socketio.emit('mqtt_message', data=data)
    # # #####################
    # if (name != 'Unknown'): 
    #     check_time=time.time()
    #     socketio.emit('check_time', data=check_time)
    #     socketio.emit('mqtt_message_name', data=name)
    #     socketio.emit('mqtt_message_emotion', data=emotion_text)
    # #####################
    # #****************************#test
#     if data == 'newblob':
#         for i in range(1000):
#             if i == 900:
#                 check_time=time.time()
#                 socketio.emit('check_time', data=check_time)
#                 socketio.emit('mqtt_message_name', data='name')
#                 socketio.emit('mqtt_message_emotion', data='emotion_text')


##############################################DEACTIVE DETECT FACE##################################################

@socketio.on('deactive_face')
def deactive_face(json_str):
    data = json_str
    if data=='deactive_face':
        ######################
        # sess = get_session()
        # clear_session()
        # sess.close()
        # sess = get_session()

        # try:
        #     del emotion_model_path
        #     del emotion_labels
        #     del emotion_classifier
        #     del kn_image
        #     del kn_face_encoding
        #     del known_face_encodings
        #     del known_face_names
        # except:
        #     pass

        # print(gc.collect()) # if it's done something you should see a number being outputted

        # # use the same config as you used to create the session
        # config = tensorflow.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 1
        # config.gpu_options.visible_device_list = "0"
        # set_session(tensorflow.Session(config=config))
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # sess = tf.Session(config=config)
        print(60*'@')
        
        ######################
        repliesmess="done_deactive"
        print(repliesmess)
        socketio.emit('done_deactive_face', data=repliesmess)


##############################################DEACTIVE DETECT FACE##################################################


##############################################ACTIVE TRAIN##################################################
@socketio.on('active_train')
# action_labels=  ['jump','kick','punch','run','sit','squat','stand','walk','wave']
######## BAI 1 ###################
# action_labels_1 =  ['none','none','none','none','sit','sit','stand','none','none']
######## BAI 2 ###################
# action_labels_2 =  ['none','none','none','none','none','squat','stand','none','none']
# ######## BAI 3 ###################
# action_labels_3 =  ['jump','none','none','none','none','none','none','none','none']
# ######## BAI 4 ###################
# action_labels_4=  ['none','none','none','run','none','none','none','none','none']
def active_train_zz(json_str):
        data = json_str
        # global action_labels
        # global my_detector
        # global multipeople_classifier,multiperson_tracker
        # if data=='active_train1':

        # ######################
                
        #         action_labels =  ['none','none','none','none','sit','sit','stand','none','none']
                
        #         my_detector = SkeletonDetector(OpenPose_MODEL, image_size)
                
        #         multipeople_classifier,multiperson_tracker=multi(LOAD_MODEL_PATH,action_labels)
        #         print(80*'*')
        #         ######################
        #         repliesmess="done_active"
        #         print(repliesmess)
                
        #         socketio.emit('done_active_train', data=repliesmess)
        #         socketio.emit('done_select_td', data='1')
        # if data=='active_train2':
        # ######################
                
                
        #         action_labels =  ['none','none','none','none','none','squat','stand','none','none']
               
        #         my_detector = SkeletonDetector(OpenPose_MODEL, image_size)
                
        #         multipeople_classifier,multiperson_tracker=multi(LOAD_MODEL_PATH,action_labels)
        #         print(80*'*')
        #         ######################
        #         repliesmess="done_active"
        #         print(repliesmess)
                
        #         socketio.emit('done_active_train', data=repliesmess)
        #         socketio.emit('done_select_td', data='2')
        # if data=='active_train3':
        # ######################
                
        #         action_labels =  ['jump','none','none','none','none','none','none','none','none']
               
        #         my_detector = SkeletonDetector(OpenPose_MODEL, image_size)
                
        #         multipeople_classifier,multiperson_tracker=multi(LOAD_MODEL_PATH,action_labels)
        #         print(80*'*')
        #         ######################
        #         repliesmess="done_active"
        #         print(repliesmess)
               
        #         socketio.emit('done_active_train', data=repliesmess)
        #         socketio.emit('done_select_td', data='3')
        # if data=='active_train4':
        # ######################
        #         action_labels =  ['none','none','none','run','none','none','none','none','none']
        #         my_detector = SkeletonDetector(OpenPose_MODEL, image_size)
        #         multipeople_classifier,multiperson_tracker=multi(LOAD_MODEL_PATH,action_labels)
        #         print(80*'*')
        #         ######################
        #         repliesmess="done_active"
        #         print(repliesmess)
        print(80*'*')     
        socketio.emit('done_active_train', data='done_active')
        socketio.emit('done_select_td', data='4')


# @socketio.on('select_td')
# def select_thd(json_str):
#         data = json_str
#         if data=='1':
#                 print('bai the duc 1')
#                 socketio.emit('done_active_train', data='done_active')
#                 socketio.emit('done_select_td', data='1')
#         if data=='2':
#                 print('bai the duc 2')
#                 socketio.emit('done_active_train', data='done_active')
#                 socketio.emit('done_select_td', data='2')
#         if data=='3':
#                 print('bai the duc 3')
#                 socketio.emit('done_active_train', data='done_active')
#                 socketio.emit('done_select_td', data='3')
#         if data=='4':
#                 print('bai the duc 4')
#                 socketio.emit('done_active_train', data='done_active')
#                 socketio.emit('done_select_td', data='4')

@socketio.on('publish_train')     # send mess
def handle_publish_train(json_str):
    data = json_str
    ######################
    # image_data = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
    # ######################
    # #*******************************************************************************************************#
    # #*******************************************************************************************************#
    # ######################
    # count_human,label,file_send=code_main(my_detector,multipeople_classifier,multiperson_tracker,image_data)
    # ######################
    # #*******************************************************************************************************#
    # ######################
    # socketio.emit('mqtt_message', data=file_send)
    # print(2*'count_human',count_human)
    # if count_human!=0:
    #     if label!='none':
    #             socketio.emit('label_human', data=label)
    #     check_time=time.time()
    #     socketio.emit('check_time_train', data=check_time)
    # if count_human=='none':
    # #     time.sleep(20)
    #     check_time=time.time()
    #     socketio.emit('check_time_canhbao', data=check_time)
    #     print('GUI CANH BAO KHONG THAY NGUOI GUI CANH BAO')
    ######################

    if data == 'newblob':
        for i in range(1000):
            if i == 900:
                check_time=time.time()
                socketio.emit('label_human', data=str(randrange(10)))
                socketio.emit('check_time_train', data=check_time)

##############################################ACTIVE TRAIN##################################################

##############################################DEACTIVE DETECT TRAIN##################################################

@socketio.on('de_active_train')
def deactive_train(json_str):
    data = json_str
    if data=='deactive_train':
        # ######################
        # sess = get_session()
        # clear_session()
        # sess.close()
        # sess = get_session()

        # try:
        #     del my_detector
        #     del multipeople_classifier
        #     del multiperson_tracker
            
        # except:
        #     pass

        # print(gc.collect()) # if it's done something you should see a number being outputted

        # # use the same config as you used to create the session
        # config = tensorflow.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 1
        # config.gpu_options.visible_device_list = "0"
        # set_session(tensorflow.Session(config=config))
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # sess = tf.Session(config=config)
        ######################
        print(60*'k')
        repliesmess="done_deactive"
        print(repliesmess)
        socketio.emit('done_deactive_train', data=repliesmess)


##############################################DEACTIVE DETECT TRAIN##################################################
# ######################
@socketio.on('sendmess')
def handle_sendmess(json_str):
    repliesmess=""
    data = json_str
    print("+++++++++")
    print(data,type(data))
    # repliesmess=response(data)
    repliesmess=data
    socketio.emit('replieszzz', data=repliesmess)
######################    
######################  


if __name__ == '__main__':
    #socketio.run(app, host='127.0.0.1', port=5000, use_reloader=True, debug=True)
    socketio.run(app, host='0.0.0.0',port=5000,use_reloader=False, debug = True,certfile="cert.pem", keyfile="key.pem")

