
from __future__ import print_function
from tkinter import *
import tkinter as tk
from tkinter import messagebox
import pandas as pd

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
from imutils import paths
import pickle
import numpy as np
import tensorflow as tf
import time
# from modulearc.evaluations import get_val_data, perform_val
# from modulearc.models import ArcFaceModel
# from modulearc.utils import set_memory_growth, load_yaml, l2_norm
from torch_mtcnn import detect_faces
from PIL import Image
from PIL import ImageDraw
from imutils.video import VideoStream
import imutils

class App(Frame):
    check_new_window = False

    def finger(self):
        os.system("python runphoto.py --o output")



    def popup(self):
        # tạo 1 popup
        id = None
        name = None
        if self.check_new_window == False:
            self.new_window3 = tk.Toplevel()
            self.new_window3.title("Training")
            self.new_window3.geometry('1000x700')
            self.new_window3.resizable(width=FALSE, height=FALSE)

            self.train_label = tk.Label(self.new_window3, text="DATA COMPARE & TRAINING")
            self.train_label.place(width=200, height=50, x=600, y=200)
            #nhap id va ten
            self.idtrain_label = tk.Label(self.new_window3, text="Enter ID")
            self.idtrain_label.place(width=100 , height=10, x=550, y=300 )
            self.nametrain_label = tk.Label(self.new_window3, text="Enter Name")
            self.nametrain_label.place(width=100, height=10, x=550, y=340)
            self.idtrain_entry = tk.Entry(self.new_window3, width=200)
            self.idtrain_entry.place(width=150 , height=20, x=650, y=300)
            self.nametrain_entry = tk.Entry(self.new_window3, width=200)
            self.nametrain_entry.place(width=150, height=20, x=650, y=340)
            self.get_id = tk.Button(self.new_window3, text="Get ID", command =self.id_name_get)
            self.get_id.place(width=100, height=60, x=870, y=300)
            #3 button training
            self.crop_img = tk.Button(self.new_window3, text="CropImage", command=self.crop_image)
            self.crop_img.place(width=300, height=60, x=550, y= 420)
            self.train_img = tk.Button(self.new_window3, text="Training", command=self.training_img)
            self.train_img.place(width=300, height=60, x=550, y=500)
            self.arcface = tk.Button(self.new_window3, text="Show Face", command=self.arc_face)
            self.arcface.place(width=300, height=60, x=550, y=580)
            self.check_new_window = True
            self.new_window3.protocol("WM_DELETE_WINDOW", self.new_window_closing)
        else:
            pass

    #def arc_face(self):


    #def training_img(self):

    def crop_image(self):
        file_obj2 = open('name_id.txt', 'r+')
        name = file_obj2.read()
        print(name)
        name = '{}'.format(name)
        # Parent Directory path
        parent_dir = r"C:\Users\admin\PycharmProjects\DoAn\Face_main\data\student"

        # Path
        path = os.path.join(parent_dir, name)
        mode = 0o666
        os.mkdir(path, mode)
        # Create the directory
        # 'GeeksForGeeks' in

        cam = VideoStream(src=0).start()
        time.sleep(2.0)
        skip_frame = 25
        total = 0
        k = 1
        while True:
            fr = cam.read()
            frame = fr.copy()
            frame = cv2.resize(frame, (250, 250))
            # frame=imutils.resize(frame, widt))
            total += 2
            if skip_frame % total == 0:
                continue
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fr = Image.fromarray(fr)
            fr = fr.resize((250, 250))
            start = time.time()
            try:
                bounding_boxes, landmarks = detect_faces(fr)
                if bounding_boxes is not None:
                    b = bounding_boxes = list(map(int, bounding_boxes[0]))
                    c = bounding_boxes[4]
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    cv2.imwrite(r'C:\Users\admin\PycharmProjects\DoAn\Face_main\data\student\{}\{}.jpg'.format(name, k), frame)
                    if k == 20:
                        break
                    k += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.putText(fr, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    end = time.time()
                    FPS = 1 / (end - start)
                    con = "{}".format(int(FPS))
                    if c * 100 > 90.0:
                        cv2.putText(frame, "{:.2f}".format(c), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    cv2.putText(frame, con, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print("infer time:{}".format(end - start))
            except IndexError:
                print("file not found")
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
    def id_name_get(self):
        self.name = self.nametrain_entry.get()
        self.idi = self.idtrain_entry.get()
        file_obj=open('name_id.txt', 'w+')
        x=file_obj.write(str(self.name))
        y=file_obj.write(self.idi)
        #print(self.name , self.idi)


    #def take_data(self):

    def new_window_closing(self):
        self.check_new_window = False
        self.new_window3.destroy()
    def face_training(self):
        os.system("py cropimage.py --o output")
    def __init__(self, master):
        super().__init__(master)

        self.fg_button = tk.Button(self, text="Finger Gesture + Face Recognition", command=self.finger, bg= "#faebd7"	)
        self.face_button = tk.Button(self, text="Face Recognition", command=self.popup, bg= "#faebd7"  )

        master.bind("<Configure>", self.placeGUI)

    def placeGUI(self, e):
        objw = self.winfo_width()
        objh = self.winfo_height()
        self.fg_button.place(x=100, y=100, width=int(objw / 3 - 40), height=int(objh / 4))
        self.face_button.place(x=100, y=int(objh * 2 / 3) - 20, width=int(objw / 3 - 40), height=int(objh / 4))

class ProcessData():

    def __init__(self):
        self.pw_trong_he_thong = 0
        self.check = False

    def check_password(self, user, pw_nhap_vao):
        data = pd.read_csv("user.csv")
        self.pw_trong_he_thong = data[data["account"] == user].password.item()
        if(str(self.pw_trong_he_thong) == pw_nhap_vao):
            self.check = True
        else:
            self.check = False

def dang_nhap():
    name_dn = name_entry.get()
    password_dn = passw_entry.get()

    print(f"The name is: {name_dn}")
    print(f"The password is: {password_dn}")
    try:
        process_data.check_password(user=name_dn, pw_nhap_vao=password_dn)
        print(process_data.check)
        if process_data.check:
            hi = App(root)
            hi.place(relwidth=1, relheight=1)
        else:
            admin_check.configure(text = "Nhập sai tên đăng nhập hoặc mật khẩu. Mời nhập lại!")
    except ValueError:
        admin_check.configure(text="Nhập sai tên đăng nhập hoặc mật khẩu. Mời nhập lại!")

root = tk.Tk()
root.title("Giao diện chính")
root.minsize(500, 500)

process_data = ProcessData()
# setting the windows size
scr_w = int(root.winfo_screenwidth()*0.5)
scr_h = int(root.winfo_screenheight()*0.5)
root.geometry(f"{scr_w}x{scr_h}+{int(root.winfo_screenwidth()/2-scr_w/2)}+{int(root.winfo_screenheight()/2-scr_h/2)}")



name_label = tk.Label(root, text='Username', font=('calibre', 10, 'bold'))

name_entry = tk.Entry(root, font=('calibre', 10, 'normal'))
name_entry.focus()
name_entry.insert(0, "nht")
passw_label = tk.Label(root, text='Password', font=('calibre', 10, 'bold'))

passw_entry = tk.Entry(root, font=('calibre', 10, 'normal'), show='*')
passw_entry.insert(0, "123")
admin_check = tk.Label(root, text = "Mời nhập tài khoản:", font=('calibre', 10, 'bold'))

#def Close():
#    root.wm_
# Button that will call the submit function
sub_btn = tk.Button(root, text='Submit', command=dang_nhap)

# placing the label and entry in
# the required position using grid
# method
name_label.grid(row=0, column=0)
name_entry.grid(row=0, column=1)
passw_label.grid(row=1, column=0)
passw_entry.grid(row=1, column=1)
admin_check.grid(row=3, column=1)
sub_btn.grid(row=2, column=1)

# performing an infinite loop
# for the window to display

root.mainloop()
# Closing Tkinter window forcefully.



