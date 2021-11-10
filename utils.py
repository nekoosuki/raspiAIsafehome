import numpy as np
from constant import *

class data_package:
    def __init__(self, payload, loadtype):
        self.load = payload
        self.type = loadtype
    
class box:
    def __init__(self, box:list):
        assert np.size(box) == 4
        self.ymin = box[0]
        self.xmin = box[1]
        self.ymax = box[2]
        self.xmax = box[3]


class I_U_cal:
    def inter(b1: box, b2: box) -> int:
        if b1.xmax <= b2.xmin or b1.ymax <= b2.ymin or b1.ymin >= b2.ymax or b1.xmin >= b2.xmax:
            return 0
        h = np.min([b1.ymax-b2.ymin, b2.ymax-b1.ymin])
        w = np.min([b1.xmax-b2.xmin, b2.xmax-b1.xmin])
        return h*w

    def union(b1: box, b2: box) -> int:
        a1 = (b1.xmax-b1.xmin)*(b1.ymax-b1.ymin)
        a2 = (b2.xmax-b2.xmin)*(b2.ymax-b2.ymin)
        a = I_U_cal.inter(b1, b2)
        return a1+a2-a

    def IOU(b1:box,b2:box)->float:
        return I_U_cal.inter(b1,b2)/I_U_cal.union(b1,b2)

class warning_gen:
    def __init__(self, iou_thre:int)->None:
        self.opwindow = None
        self.move = None
        self.detected = False
        self.iou = iou_thre

    def update_window(self, opwindow:list)->None:
        self.opwindow = opwindow

    def update_move(self, move:list)->None:
        if move is not None:
            self.detected = True
        self.move = move

    def close_window(self)->bool:
        if self.opwindow is None or self.move is None:
            return False
        for w in self.opwindow:
            bw = box(w)
            for m in self.move:
                bm = box(m)
                iou = I_U_cal.IOU(bw,bm)
                if iou > self.iou:
                    return True
        return False

    def is_disapr(self)->bool:
        if self.detected and self.move is None:
            self.detected = False
            return True
        return False
