import cv2
import imutils
from torch_mtcnn import detect_faces
from PIL import Image
from PIL import ImageDraw
import os
import time
import numpy as np
from imutils.video import VideoStream
import os
# get the script path



file_obj2= open('name_id.txt', 'r+')
name = file_obj2.read()
print(name)
name='{}'.format(name)
# Parent Directory path
parent_dir = r"C:\Users\admin\PycharmProjects\DoAn\take_data\test"

# Path
path = os.path.join(parent_dir, name)
mode=0o666
os.mkdir(path, mode)
# Create the directory
# 'GeeksForGeeks' in





cam  = VideoStream(src=0).start()
time.sleep(2.0)
skip_frame=20
total=0
k=1
while True:
    fr = cam.read()
    frame=fr.copy()
    frame=cv2.resize(frame, (460, 259))
        #frame=imutils.resize(frame, widt))
    total+=1
    if skip_frame % total ==0:
        continue
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fr=Image.fromarray(fr)
    fr=fr.resize((460, 259))
    start=time.time()
    try:
        bounding_boxes, landmarks = detect_faces(fr)
        if bounding_boxes is not None:
            b=bounding_boxes = list(map(int, bounding_boxes[0]))
            c=bounding_boxes[4]
            x1,y1,x2,y2=b[0], b[1], b[2], b[3]
            crop=frame[y1-6 :y2+50, x1-6:x2+50]
        #crop = cv2.resize(crop, (112, 112))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(r'C:\Users\admin\PycharmProjects\DoAn\take_data\test\{}\{}.jpg'.format(name, k), frame)
            time.sleep(1)
            if k==20:
                break
            k+=1
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
            #cv2.putText(fr, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            end=time.time()
            FPS=1/(end-start)
            con="{}".format(int(FPS))
            if c*100>90.0:
               cv2.putText(frame, "{:.2f}".format(c), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(frame, con,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2  )
            print("infer time:{}".format(end-start))
    except IndexError:
        print("file not found")
    cv2.imshow("frame", frame)
    cv2.waitKey(1)




