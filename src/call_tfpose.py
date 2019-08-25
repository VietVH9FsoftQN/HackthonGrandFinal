import code_tfpose
from code_tfpose import *
my_detector = SkeletonDetector(OpenPose_MODEL, image_size)
multipeople_classifier,multiperson_tracker=multi(LOAD_MODEL_PATH,action_labels)

import os
inputpath = 'frame_train2'
files = [f for f in os.listdir(inputpath)]
files.sort(key = lambda x: int(x[5:-4]))
files
for i in range(len(files)):
        image = cv2.imread(inputpath + '/' + files[i])
        count_human,label,jpeg,image_disp=code_main(my_detector,multipeople_classifier,multiperson_tracker,image)
        with open('test.txt', 'a') as f:
            f.write(label + '\n')
        file_name="save_image/" + str(i) + ".jpg"
        cv2.imwrite(file_name, image_disp)