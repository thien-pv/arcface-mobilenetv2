#!/usr/bin/python -tt
# Original code
#https://github.com/deepinsight/insightface/tree/master/model_zoo
#https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/align/face_align.py
import torch
import argparse
import os
from imutils.video import VideoStream
import torch.utils.data as data
#import torchvision.datasets as datasets
import torch.nn.functional as F
#import torchvision.transforms as transforms
from backbone import Backbone
#mtcnn
import time
from torch_mtcnn import detect_faces
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
from align.align_trans import (
    get_reference_facial_points,
    warp_and_crop_face,
)
import math
from PIL import Image
#from tqdm import tqdm
from imutils import paths
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--crop_size",
    help="specify size of aligned faces",
    default=112,
    choices=[112, 224],
    type=int,
)
args = parser.parse_args()
crop_size = args.crop_size
scale = crop_size / 112.0
reference = get_reference_facial_points(default_square=True) * scale
model_root=r"/home/lab506/Desktop/doaanan/Face_main/checkpoint/backbone_ir50_ms1m_epoch120.pth"
    # load backbone weigths from a checkpoint
le = pickle.loads(open("./output/le.pickle", "rb").read())
recognizer = pickle.loads(open("./output/recognizer.pickle", "rb").read())
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
backbone = Backbone(input_size=[112, 112], num_layers=50)
backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
backbone.to(device)
backbone.eval()
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1200,
    display_width=480,
    display_height=300,
    framerate=28,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d,      		framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def show_camera():
    print(gstreamer_pipeline(flip_method=2))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    while True:
               
                          _, fr = video_capture.read()
                      
                          frame=fr.copy()
                          frame=cv2.resize(frame, (480 , 300))
                         
                          fr=Image.fromarray(fr)
                  
                          fr=fr.resize((480 , 300))

              
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
                               
                                  img_warped  = cv2.cvtColor(warped_face , cv2.COLOR_BGR2RGB)
                                  
                      

                                  img = np.transpose(img_warped, (2, 0, 1))
                                  img = torch.from_numpy(img).unsqueeze(0).float()
                                  img.div_(255).sub_(0.5).div_(0.5)

                                  embeddings = np.zeros([1,512])
                                  embeddings= F.normalize(backbone(img.to(device))).cpu().detach().numpy()

                                  preds = recognizer.predict_proba(embeddings)[0]
                                  j = np.argmax(preds)
                                  proba = preds[j]
                                  name = le.classes_[j]
                                  text = "{}: {:.2f}%".format(name, proba * 100)
                                  cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                                  y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                                  cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
                 
                                  end=time.time()
                                  FPS=1/(end-start)
                                  print("MTCNN_ARCFACE.....Score:{:.2f}".format(proba * 100))
                                  con ="{}".format(int(FPS))
                                  print("infer time:{}".format(end-start))
                                  cv2.putText(frame, con,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2  )

                          except IndexError:
                                    print("file not found")

                          cv2.imshow("frame", frame)
                          keyCode = cv2.waitKey(10) & 0xFF
                          if keyCode == 27 or keyCode == ord('q'):
                              break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera()

