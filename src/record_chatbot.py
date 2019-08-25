import random
import time
import os
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from nltk.translate.bleu_score import sentence_bleu

#############################
input_file = open("input_chatbot.txt", "r", encoding = "utf8")
output_file = open("output_chatbot.txt","r",encoding = "utf8")
input_lines = []
output_lines = []

# SỬA TÊN FILE ÂM THANH VỪA RECORD ĐƯỢC CHO ĐÚNG ĐƯỜNG DẪN#
#file = "input_chatbot.wav"


#############################

for line in input_file:
    input_lines.append(line[:-1])

for line in output_file:
    output_lines.append(line)

output_file.close()
input_file.close()

# Điền path vào
import random
import time
import os
import speech_recognition as sr

def speech_to_text(file):    
    # Chuyển file âm thanh đã lưu thành dạng text
    
    print("NGU")
    r = sr.Recognizer()
    print("NGU")
    with sr.WavFile(file) as source:
        print("NGU")
        audio = r.record(source)
        print("NGU")
        output = r.recognize_google(audio)
        print("NGU")
        print(output)
#    except:
#        output = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        
    os.remove(file)
    
    print(output)
    return output 

def chose_answer(input_text): 
    # Chọn câu trả lời cho chat bot
    
    arr = [sentence_bleu([candidates],input_text) for candidates in input_lines]
    print("Score: ",arr)
    output_index = arr.index(max(arr))
    
    output = output_lines[output_index+1]
         
    return output    
def text_to_speech(theText,name):
    # Chuyển sang file âm thanh phát trên ohmni
    try:
        os.remove("./static/audio/output_chatbot.mp3")
    except:
        print("Dupplicate removed")
    
    tts = gTTS(text=theText, lang='en')

    tts.save("./static/audio/"+name) ######################
    print("File saved!")
import datetime    
def text_to_speech_1(theText):
    # Chuyển sang file âm thanh phát trên ohmni

    name = "output"+datetime.datetime.now().strftime("%s")+".mp3"
    tts = gTTS(text=theText, lang='en')

    tts.save("./static/audio/"+name) ######################
    print("File saved!")
    return name
    


# file là cái record từ người gọi con robot
def speech_to_speech(file):
    # Code duy nhất cần chạy sau khi lưu file âm thanh xong
    
    input_text = speech_to_text(file)
    answer = chose_answer(input_text)
    print(answer)
    name = "output"+datetime.datetime.now().strftime("%s")+".mp3"
    text_to_speech(answer,name)
    return name