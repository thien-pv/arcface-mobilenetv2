from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
# code finger counter
import gettext
import mediapipe as mp
import os
import time

###
Hands = mp.solutions.hands
Draw = mp.solutions.drawing_utils


class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = Hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
                                 min_tracking_confidence=min_tracking_confidence)

    def findHandLandMarks(self, image, handNumber=0, draw=False):
        originalImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # mediapipe needs RGB

        results = self.hands.process(image)
        landMarkList = []

        def get_label(index, hand, results):
            output = None
            for idx, classification in enumerate(results.multi_handedness):
                if classification.classification[0].index == index:
                    # Process results
                    label = classification.classification[0].label
                    score = classification.classification[0].score
                    text = '{} {}'.format(label, round(score, 2))

                    output = text

            return output

        text = None
        if results.multi_hand_landmarks:  # returns None if hand is not found
            hand = results.multi_hand_landmarks[
                handNumber]  # results.multi_hand_landmarks returns landMarks for all the hands

            for id, landMark in enumerate(hand.landmark):
                # landMark holds x,y,z ratios of single landmark
                imgH, imgW, imgC = originalImage.shape  # height, width, channel for image
                xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                landMarkList.append([id, xPos, yPos])

                if get_label(id, landMark, results):
                    text = get_label(id, landMark, results)

            if draw:
                Draw.draw_landmarks(originalImage, hand, Hands.HAND_CONNECTIONS)

        # print(text)

        return landMarkList


handDetector = HandDetector()


class PhotoBoothApp:
    def __init__(self, vs, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        #self.stopEvent = None
        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None
        # create a button, that when pressed, will take the current
        # frame and save it to file

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
        # set a callback to handle when the window is closed
        self.root.wm_title("Hand Gesture ")
        self.root.geometry("1000x1000")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoLoop(self):
        check1 = True
        check2 = True

        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        pTime = 0
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                self.frame = self.vs.read()
                self.frame = cv2.resize(self.frame, (460, 259))
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                handLandmarks = handDetector.findHandLandMarks(image=image, draw=True)
                self.count = 0
                if (len(handLandmarks) != 0):

                    if (((handLandmarks[4][2] - handLandmarks[0][2]) ** 2 + (handLandmarks[4][1] - handLandmarks[0][1]) ** 2) > (
                            (handLandmarks[2][2] - handLandmarks[0][2]) ** 2 + (handLandmarks[2][1] - handLandmarks[0][1]) ** 2)
                            and ((handLandmarks[4][2] - handLandmarks[17][2]) ** 2 + (
                                    handLandmarks[4][1] - handLandmarks[17][1]) ** 2) > (
                                    (handLandmarks[2][2] - handLandmarks[17][2]) ** 2 + (
                                    handLandmarks[2][1] - handLandmarks[17][1]) ** 2)):  # Thumb fqinger
                            self.count = self.count + 1
                    if ((handLandmarks[8][2] - handLandmarks[0][2]) ** 2 + (handLandmarks[8][1] - handLandmarks[0][1]) ** 2) > (
                            (handLandmarks[6][2] - handLandmarks[0][2]) ** 2 + (
                            handLandmarks[6][1] - handLandmarks[0][1]) ** 2):  # Index finger
                            self.count = self.count + 1
                    if ((handLandmarks[12][2] - handLandmarks[0][2]) ** 2 + (handLandmarks[12][1] - handLandmarks[0][1]) ** 2) > (
                            (handLandmarks[10][2] - handLandmarks[0][2]) ** 2 + (
                            handLandmarks[10][1] - handLandmarks[0][1]) ** 2):  # Middle finger
                            self.count = self.count + 1
                    if ((handLandmarks[16][2] - handLandmarks[0][2]) ** 2 + (handLandmarks[16][1] - handLandmarks[0][1]) ** 2) > (
                            (handLandmarks[14][2] - handLandmarks[0][2]) ** 2 + (
                            handLandmarks[14][1] - handLandmarks[0][1]) ** 2):  # Ring finger
                            self.count = self.count + 1
                    if ((handLandmarks[20][2] - handLandmarks[0][2]) ** 2 + (handLandmarks[20][1] - handLandmarks[0][1]) ** 2) > (
                            (handLandmarks[18][2] - handLandmarks[0][2]) ** 2 + (
                            handLandmarks[18][1] - handLandmarks[0][1]) ** 2):
                            self.count = self.count + 1
        # x=handLandmarks[4][1]
        # y=handLandmarks[4][2]
                cv2.putText(image, str(self.count), (45, 205), cv2.FONT_HERSHEY_SIMPLEX, 3, (240, 120, 0), 10)



                if self.count == 1:
                    # a = tki.Label(self.root, text="den sang")
                    # a.pack()
                    self.img = ImageTk.PhotoImage(Image.open("lighton.jpg"))

                    panel = tki.Label(self.root, image=self.img)
                    panel.place(width=200, height = 200, x = 50 , y = 50)


                elif self.count == 2:

                    self.img = ImageTk.PhotoImage(Image.open("lightoff.jpg"))

                    panel = tki.Label(self.root, image=self.img)
                    panel.place(width=200, height=200, x=50, y=50)

                if self.count == 3:

                    self.img = ImageTk.PhotoImage(Image.open("fanon.jpg"))

                    panel = tki.Label(self.root, image=self.img)
                    panel.place(width=200, height=200, x=750, y=50)
                elif self.count == 4:

                    self.img = ImageTk.PhotoImage(Image.open("fanoff.jpg"))

                    panel = tki.Label(self.root, image=self.img)
                    panel.place(width=200, height=200, x=750, y=50)

                #print(check1, check2)


                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(image, f"FPS: {int(fps)}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
                # cv2.putText(image, "x="+str(x), (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                # cv2.putText(image, "y="+str(y), (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                #img=image.copy()
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(anchor="center", padx=10, pady=10)

            # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image



        except RuntimeError as e:
           print("[INFO] caught a RuntimeError")


    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()

