import numpy as np
from constant import *
import time


class data_package:
    def __init__(self, payload, loadtype):
        self.load = payload
        self.type = loadtype


class box:
    def __init__(self, box: list):
        assert np.size(box) == 4
        self.ymin = box[0]
        self.xmin = box[1]
        self.ymax = box[2]
        self.xmax = box[3]


class I_U_cal:
    "    计算 box 交集和并集\n"
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

    def IOU(b1: box, b2: box) -> float:
        return I_U_cal.inter(b1, b2)/I_U_cal.union(b1, b2)


class warning_gen:
    def __init__(self, iou_thre: int) -> None:
        "    获取运动物体靠近窗户或消失警告\n\n    参数\n    ----------\n    iou_thre : 判断物体靠近窗户的iou阈值\n"
        self.opwindow = None
        self.move = None
        self.detected = False
        self.iou = iou_thre

    def update_window(self, opwindow: list) -> None:
        "    更新 window\n\n    参数\n    ----------\n    opwindow : 窗户 box 的列表\n\n    形状 : (,4)\n"
        self.opwindow = opwindow

    def update_move(self, move: list) -> None:
        "    更新 move\n\n    参数\n    ----------\n    move : 运动 box 的列表\n\n    形状 : (,4)\n"
        if move is not None:
            self.detected = True
        self.move = move

    def close_window(self) -> bool:
        "    根据 iou_thre 判断 window 和 move 列表中是否有任一对 box 的 IOU 大于等于 iouthre\n\n    用于判断是否有运动物体靠近窗户\n\n    返回\n    ----------\n    False : 从未更新过 window 和 move 或没有 IOU 大于 iouthre\n\n    True : 任一对 IOU 大于 iouthre\n"
        if self.opwindow is None or self.move is None:
            return False
        for w in self.opwindow:
            bw = box(w)
            for m in self.move:
                bm = box(m)
                iou = I_U_cal.IOU(bw, bm)
                if iou > self.iou:
                    return True
        return False

    def is_disapr(self) -> bool:
        "    判断运动物体是否消失\n\n    返回\n    ----------\n    False : 运动物体从未出现或未消失\n\n    True : 运动物体消失\n\n    注意\n    ----------\n    该函数在运动物体一次消失间隔中只会返回一次 True\n"
        if self.detected and self.move is None:
            self.detected = False
            return True
        return False


class Counter:
    def __init__(self) -> None:
        "    根据标签进行并行计时\n"
        self.timer = {}

    def start(self, label: str) -> None:
        "    启动计时器\n\n    参数\n    ----------\n    label : 计时器标签\n"
        self.timer[label] = time.perf_counter()

    def end(self, label: str, useFPS: bool = True) -> str:
        "    返回距离调用start()的时间\n\n    参数\n    ----------\n    label : 计时器标签\n\n    useFPS : 如果为 True 以 FPS 格式返回，否则以秒格式返回\n\n    返回\n    ----------\n    描述计时结果的字符串\n\n    注意\n    ----------\n    调用这个方法不会重置计时器\n    要重置计时器，请重新调用start()\n"
        return '{}FPS:{}'.format(label, 1/(time.perf_counter()-self.timer[label])) if useFPS else '{}timing:{}'.format(label, time.perf_counter()-self.timer[label])
