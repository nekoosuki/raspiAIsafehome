import cv2
import subprocess
import numpy as np
import mediapipe as mp
from constant import *


class VideoCap:
    labels = {OBJ_IDX_CLOSEWINDOW: 'windowclose',
              OBJ_IDX_OPENWINDOW: 'windowopen'}

    def __init__(self, w: int, h: int, url: str) -> None:
        "视频源管理和绘图推流操作\n\n    参数\n    ----------\n    w : 画面的宽度，该参数在使用视频时无效\n\n    h : 画面的高度，该参数在使用视频时无效\n\n    url : rtmp推流地址\n"
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
        "    从源读一帧数据，存入对象内部并返回\n\n    注意\n    ---------\n    该函数返回引用，对返回数组的操作会影响对象内部数据。如果不想影响，请使用VideoCap.read().copy()\n"
        succ, self.frame = self.cap.read()
        if succ:
            return self.frame
        else:
            return np.array([])

    def draw_bbox(self, pdt_l: list):
        "    对对象存储的帧绘制带标签和置信度的 bbox\n\n    参数\n    ----------\n    pdt_l : [box[ymin, xmin, ymax, xmax], class, score] 组成的二维列表\n"
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
        "    对对象存储的帧绘制 bbox\n\n    参数\n    ----------\n    diffbox : [ymin, xmin, ymax, xmax] 组成的二维列表\n"
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
        "    对对象存储的帧绘制 pose_landmark\n\n"
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(self.frame, landmarks, connections)

    def write(self, frame=None):
        "    将对象存储的帧或指定帧输出到视频文件\n\n    参数\n    ----------\n    frame : None 或 BGR格式的帧"
        if frame is None:
            self.out.write(self.frame)
        else:
            self.out.write(frame)

    def release(self):
        "    释放全部资源\n"
        self.cap.release()
        self.out.release()
        self.pipe.terminate()
