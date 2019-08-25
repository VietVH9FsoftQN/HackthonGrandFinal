import numpy as np
import cv2
import sys, os, time, argparse, logging
import simplejson
import argparse
import math
import mylib.io as myio
from mylib.displays import drawActionResult
import mylib.funcs as myfunc
import mylib.feature_proc as myproc 
from mylib.action_classifier import ClassifierOnlineTest
from mylib.action_classifier import * # Import sklearn related libraries
image_size = "240x208"
OpenPose_MODEL = ["mobilenet_thin", "cmu"][1]
LOAD_MODEL_PATH = "../model/trained_classifier.pickle"

action_labels=  ['jump','kick','punch','run','sit','squat','stand','walk','wave']
# Openpose include files and configs ==============================================================
sys.path.append("githubs/tf-pose-estimation")
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common
# ---- For tf 1.13.1, The following setting is needed
import tensorflow as tf
from tensorflow import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# If GPU memory is small, modify the MAX_FRACTION_OF_GPU_TO_USE
MAX_FRACTION_OF_GPU_TO_USE = 0.5
config.gpu_options.per_process_gpu_memory_fraction=MAX_FRACTION_OF_GPU_TO_USE
# Openpose Human pose detection ==============================================================
class SkeletonDetector(object):
    def __init__(self, model=None, image_size=None):
        
        if model is None:
            model = "cmu"

        if image_size is None:
            image_size = "432x368" # 7 fps

        models = set({"mobilenet_thin", "cmu"})
        self.model = model if model in models else "mobilenet_thin"
        self.resize_out_ratio = 4.0

        w, h = model_wh(image_size)
        if w == 0 or h == 0:
            e = TfPoseEstimator(
                    get_graph_path(self.model),
                    target_size=(432, 368),
                    tf_config=config)
        else:
            e = TfPoseEstimator(
                get_graph_path(self.model), 
                target_size=(w, h),
                tf_config=config)

        # self.args = args
        self.w, self.h = w, h
        self.e = e
        self.fps_time = time.time()
        self.cnt_image = 0

    def detect(self, image):
        self.cnt_image += 1
        if self.cnt_image == 1:
            self.image_h = image.shape[0]
            self.image_w = image.shape[1]
            self.scale_y = 1.0 * self.image_h / self.image_w
        t = time.time()

        # Inference
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                #   upsample_size=self.args.resize_out_ratio)
                                  upsample_size=self.resize_out_ratio)

        # Print result and time cost
        elapsed = time.time() - t
#         logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans
    
    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
    def humans_to_skelsList(self, humans, scale_y = None): # get (x, y * scale_y)
        if scale_y is None:
            scale_y = self.scale_y
        skeletons = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(18*2)
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[2*idx]=body_part.x
                skeleton[2*idx+1]=body_part.y * scale_y
            skeletons.append(skeleton)
        return skeletons, scale_y
    


# ==============================================================

def add_white_region_to_left_of_image(image_disp):
    r, c, d = image_disp.shape
    blank = 255 + np.zeros((r, int(c/4), d), np.uint8)
    image_disp = np.hstack((blank, image_disp))
    return image_disp

def remove_skeletons_with_few_joints(skeletons):
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x!=0])
        num_leg_joints = len([x for x in px[-6:] if x!=0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 2: 
            good_skeletons.append(skeleton) # add this skeleton only when all requirements are satisfied
    return good_skeletons

class MultiPersonClassifier(object):
    def __init__(self, LOAD_MODEL_PATH, action_labels):
        self.create_classifier = lambda human_id: ClassifierOnlineTest(
            LOAD_MODEL_PATH, action_types = action_labels, human_id=human_id)
        self.dict_id2clf = {} # human id -> classifier of this person

    def classify(self, dict_id2skeleton):

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():
            
            if id not in self.dict_id2clf: # add this new person
                self.dict_id2clf[id] = self.create_classifier(id)
            
            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton) # predict label

        return id2label

    def get(self, id):
        # type: id: int or "min"
        if len(self.dict_id2clf) == 0:
            return None 
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]

def multi(LOAD_MODEL_PATH,action_labels):
    multipeople_classifier = MultiPersonClassifier(LOAD_MODEL_PATH, action_labels)
    multiperson_tracker = myfunc.Tracker()
    return multipeople_classifier,multiperson_tracker

# multipeople_classifier,multiperson_tracker=multi(LOAD_MODEL_PATH,action_labels)
def code_main(my_detector,multipeople_classifier,multiperson_tracker,image):
    try:
        humans = my_detector.detect(image)
        count_human=len(humans)
        print('count_human',len(humans))
        skeletons, scale_y = my_detector.humans_to_skelsList(humans)
        skeletons = remove_skeletons_with_few_joints(skeletons)
        dict_id2skeleton = multiperson_tracker.track(skeletons) # int id -> np.array() skeleton
        min_id = min(dict_id2skeleton.keys())
        dict_id2label = multipeople_classifier.classify(dict_id2skeleton)
        print("prediced label is :", dict_id2label[min_id])
        label=dict_id2label[min_id]
        my_detector.draw(image, humans) # Draw all skeletons
        ith_img = 1
        if len(dict_id2skeleton):  
            for id, label in dict_id2label.items():
                skeleton = dict_id2skeleton[id]
                skeleton[1::2] = skeleton[1::2] / scale_y 
                drawActionResult(image, id, skeleton, label)
        image_disp = add_white_region_to_left_of_image(image)
        multipeople_classifier.get(id='min').draw_scores_onto_image(image_disp)
        ret, jpeg = cv2.imencode('.jpg', image_disp)
        return count_human,label,jpeg.tobytes()
    except:
        print('no_human')
        ret, jpeg = cv2.imencode('.jpg', image)
        return 0,'none',jpeg.tobytes()
