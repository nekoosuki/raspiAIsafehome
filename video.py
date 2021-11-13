import cv2
import subprocess
import numpy as np
import mediapipe as mp
from constant import *


class VideoCap:
    labels = {OBJ_IDX_CLOSEWINDOW: 'windowclose',
              OBJ_IDX_OPENWINDOW: 'windowopen'}

    def __init__(self, w: int, h: int, url: str) -> None:
        self.cap = cv2.VideoCapture('test_video.mp4')
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        # self.cap.set(cv2.CAP_PROP_CONTRAST,0.5)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, 1)
        # self.w = w
        # self.h = h
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(
            'output1.avi', fourcc, 20.0, (round(w), round(h)))

        sizeStr = f'{w}x{h}'
        command = ['ffmpeg',
                   '-y', '-an',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', sizeStr,
                   '-r', '20',
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-f', 'flv',
                   url]
        self.pipe = subprocess.Popen(
            command, shell=False, stdin=subprocess.PIPE)

    def read(self) -> np.ndarray:
        succ, self.frame = self.cap.read()
        if succ:
            return self.frame
        else:
            return np.array([])

    def draw_bbox(self, pdt_l: list):
        for e in pdt_l:
            # e[0]:box[ymin,xmin,ymax,xmax]
            # e[1]:class
            # e[2]:score
            ymin = int(e[0][0])
            xmin = int(e[0][1])
            ymax = int(e[0][2])
            xmax = int(e[0][3])
            # 矩形框
            cv2.rectangle(self.frame, (xmin, ymin),
                          (xmax, ymax), (10, 255, 0), 2)
            obj_name = self.labels[int(e[1])]
            # 标签名
            label = '%s: %d%%' % (obj_name, int(e[2]*100))
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            # 标签框
            cv2.rectangle(self.frame, (xmin, label_ymin-labelSize[1]-10), (
                xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(self.frame, label, (xmin, label_ymin-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def draw_diffbox(self, diffbox: list):
        text = 'static'
        for box in diffbox:
            ymin = box[0]
            xmin = box[1]
            ymax = box[2]
            xmax = box[3]
            cv2.rectangle(self.frame, (xmin, ymin),
                          (xmax, ymax), (0, 255, 0), 2)
            text = 'move'
        cv2.putText(self.frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def draw_pose(self, landmarks, connections):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(self.frame, landmarks, connections)

    def write(self, frame=None):
        # 写到文件
        if frame is None:
            self.out.write(self.frame)
        else:
            self.out.write(frame)
