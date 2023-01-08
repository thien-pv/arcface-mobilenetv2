# Original code
#https://github.com/deepinsight/insightface/tree/master/model_zoo
#https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/align/face_align.py
import torch
import argparse
import os
from imutils.video import VideoStream
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from Face_main.backbone import Backbone
#mtcnn
import time
from torch_mtcnn import detect_faces
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
from Face_main.align.align_trans import (
    get_reference_facial_points,
    warp_and_crop_face,
)
from PIL import Image
#from tqdm import tqdm
from imutils import paths
import pickle
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from PIL import Image
from PIL import ImageTk

import cv2 as cv
import numpy as np
import mediapipe as mp
import tkinter as tk

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
   # parser.add_argument("--width", help='cap width', type=int, default=460)
    #.add_argument("--height", help='cap height', type=int, default=259)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    arg = parser.parse_args()

    return arg
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    # if info_text == "Left":
    #     info_text = "Left: Mode High"
    # elif info_text == "Right":
    #     info_text = "Right: Light"
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text


    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image
parser = argparse.ArgumentParser()
parser.add_argument(
    "--crop_size",
    help="specify size of aligned faces",
    default=112,
    choices=[112, 224],
    type=int,
)
arg= get_args()
use_static_image_mode = arg.use_static_image_mode
min_detection_confidence = arg.min_detection_confidence
min_tracking_confidence = arg.min_tracking_confidence
args = parser.parse_args()
crop_size = args.crop_size
scale = crop_size / 112.0
reference = get_reference_facial_points(default_square=True) * scale
model_root="./Face_main/checkpoint/backbone_ir50_ms1m_epoch120.pth"
    # load backbone weigths from a checkpoint
le = pickle.loads(open("./Face_main/output/le.pickle", "rb").read())
recognizer = pickle.loads(open("./Face_main/output/recognizer.pickle", "rb").read())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone = Backbone(input_size=[112, 112], num_layers=50)
backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
backbone.to(device)
backbone.eval()
use_brect = True
    # Model load #############################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=2,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)

keypoint_classifier = KeyPointClassifier()

point_history_classifier = PointHistoryClassifier()

# Read labels ###########################################################
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
with open(
        'model/point_history_classifier/point_history_classifier_label.csv',
        encoding='utf-8-sig') as f:
    point_history_classifier_labels = csv.reader(f)
    point_history_classifier_labels = [row[0] for row in point_history_classifier_labels ]

# FPS Measurement ########################################################
cvFpsCalc = CvFpsCalc(buffer_len=10)

# Coordinate history #################################################################
history_length = 16
point_history = deque(maxlen=history_length)

# Finger gesture history ################################################
finger_gesture_history = deque(maxlen=history_length)

#  ########################################################################
mode = 0

cam  = VideoStream(src=0).start()
time.sleep(2.0)
while True:
    fr = cam.read()
    #if fr is None:ps
    fps = cvFpsCalc.get()
    #    print("no cam input")
    frame=fr.copy()
    image=fr.copy()
    frame=cv2.resize(frame, (512 , 512))

    image=cv2.resize(image, (512 , 512))

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fr=Image.fromarray(fr)
    #frame=Image.open(fr)
    fr=fr.resize((512 , 512))
    #img=cv2.imread(fr)
    #img=cv2.resize(img, (512, 512))
    start=time.time()
    bounding_boxes, landmarks = detect_faces(fr)
    try:
        if bounding_boxes is not None:
            b=bounding_boxes = list(map(int, bounding_boxes[0]))
            x1,y1,x2,y2=b[0], b[1], b[2], b[3]
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(
                np.array(frame),
                facial5points,
                reference,
                crop_size=(crop_size, crop_size),
            )
            #img_warped = Image.fromarray(warped_face)
            img_warped  = cv2.cvtColor(warped_face , cv2.COLOR_BGR2RGB)
            #img_warped=Image.open(img_warped)
            #cv2.imshow("frame", img_warped )
            #cv2.waitKey(0)
    #------------------------------
            img = np.transpose(img_warped, (2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float()
            img.div_(255).sub_(0.5).div_(0.5)
            #embeddings= F.normalize(backbone(img))
            #feat = backbone(img).numpy()
            #feat = backbone(img).detach().numpy()
            #print(feat)
            embeddings = np.zeros([1,512])
            embeddings= F.normalize(backbone(img.to(device))).cpu().detach().numpy()
            #embeds=np.array(embeddings)
            #print(embeddings)
            preds = recognizer.predict_proba(embeddings)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            text = "{}: {:.2f}%".format(name, proba * 100)
            '''cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
            #if proba*100>20:
              # cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            end=time.time()
            FPS=1/(end-start)
            print("MTCNN_ARCFACE.....Score:{:.2f}".format(proba * 100))
            #print("MTCNN_ARCFACE.....FPS:{}".format(int(FPS)))
            con="{}".format(int(FPS))
            print("infer time:{}".format(end-start))'''
            #cv2.putText(frame, con,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2  )
            #########################################################################
            if proba * 100>80:
                key = cv.waitKey(10)
                if key == 27:  # ESC
                    break
                number, mode = select_mode(key, mode)
                #image = cv.flip(image, 1)  # Mirror display
                image = cv.resize(image, dsize=None, fx=1, fy=1)
                debug_image = copy.deepcopy(image)

                # Detection implementation #############################################################
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                #  ####################################################################
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                          results.multi_handedness):
                        # Bounding box calculation
                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        # Landmark calculation
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list)
                        pre_processed_point_history_list = pre_process_point_history(
                            debug_image, point_history)
                        # Write to the dataset file
                        logging_csv(number, mode, pre_processed_landmark_list,
                                    pre_processed_point_history_list)

                        # Hand sign classification
                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        if hand_sign_id == 2:  # Point gesture
                            point_history.append(landmark_list[8])
                        else:
                            point_history.append([0, 0])

                        # Finger gesture classification
                        finger_gesture_id = 0
                        point_history_len = len(pre_processed_point_history_list)
                        if point_history_len == (history_length * 2):
                            finger_gesture_id = point_history_classifier(
                                pre_processed_point_history_list)

                        # Calculates the gesture IDs in the latest detection
                        finger_gesture_history.append(finger_gesture_id)
                        most_common_fg_id = Counter(
                            finger_gesture_history).most_common()

                        # Drawing part
                        debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                        debug_image = draw_landmarks(debug_image, landmark_list)
                        debug_image = draw_info_text(
                            debug_image,
                            brect,
                            handedness,
                            keypoint_classifier_labels[hand_sign_id],
                            point_history_classifier_labels[most_common_fg_id[0][0]],
                        )
                else:
                    point_history.append([0, 0])

                debug_image = draw_point_history(debug_image, point_history)
                debug_image = draw_info(debug_image, fps, mode, number)
                cv2.putText(debug_image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # if proba*100>20:
                # cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                end = time.time()
                FPS = 1 / (end - start)
                print("MTCNN_ARCFACE.....Score:{:.2f}".format(proba * 100))
                # print("MTCNN_ARCFACE.....FPS:{}".format(int(FPS)))
                con = "{}".format(int(FPS))
                print("infer time:{}".format(end - start))
                # Screen reflection #############################################################
                #debug_image = cv.flip(debug_image, 1)  # Mirror display
                cv2.imshow('Hand Gesture Recognition', debug_image)
            #else:
                #cv2.imshow("frame", frame)
    except IndexError:
       print("file not found")
    #cv2.imshow("frame", frame)
    #cv2.imshow('Hand Gesture Recognition', debug_image)
    cv2.waitKey(1)

