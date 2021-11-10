import cv2
import imutils
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import mediapipe as mp
from constant import *

class objDetector:
    def __init__(self, model, w, h, conf_thre):
        self.interpreter = Interpreter(model_path=model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        self.w=w
        self.h=h
        self.conf = conf_thre
    
    def detect(self, frame):
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320,320))
        input_data = np.expand_dims(frame, axis=0)
        if self.floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[
            0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[
            0]
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])[0]

        zipped = zip(boxes,classes, scores)
        ret = []
        for e in zipped:
            # e[0]:box[ymin,xmin,ymax,xmax]
            # e[1]:class
            # e[2]:score
            if e[2] > self.conf and e[2] <= 1.0:
                e[0][0] = int(max(1,e[0][0]*self.h))
                e[0][1] = int(max(1,e[0][1]*self.w))
                e[0][2] = int(min(self.h,e[0][2]*self.h))
                e[0][3] = int(min(self.w,e[0][3]*self.w))
                ret.append(e)

        return ret




class movementDetector:
    def __init__(self, frame, min_area=4000):
        #f = imutils.resize(frame, width=500)
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.comp_frame = cv2.GaussianBlur(f, (21,21),0)
        self.min_area = min_area

        

    def detect(self, frame):
        # frame = cv2.resize(frame, (320,320))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        frameDiff = cv2.absdiff(self.comp_frame, gray_frame)
        retVal, thresh = cv2.threshold(frameDiff, 20, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, (50,50), iterations=3)
        img, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        diffbox = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            x,y,w,h = cv2.boundingRect(contour)
            diffbox.append((y,x,y+h,x+w))
            #cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frameDiff, thresh, diffbox

    def reset_frame(self, frame):
        #f = imutils.resize(frame, width=500)
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.comp_frame = cv2.GaussianBlur(f, (21,21),0)




class poseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode = False,model_complexity=2)
        self.interpreter = Interpreter("model-lite.tflite")
        self.interpreter.allocate_tensors()
        
        
    def detect(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame)
        return results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS

    def pose_invoke(self, landmarks):
        x = []
        try:
            for item in landmarks.landmark:
                x.append([item.x,item.y,item.z,item.visibility])
            x = x[-22:]
        except:
            return np.array([])
        if np.size(x)!=0:
            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'],np.array(x,dtype='float32').reshape((1,22,4)))
            self.interpreter.invoke()
            r = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])[0]
            return r
        return np.array([])
            

