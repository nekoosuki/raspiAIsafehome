import threading
from video import *
from detector import *
import time
import queue
from utils import data_package as dp
from utils import *
from constant import *

rq = queue.Queue(5)
dq = queue.Queue(10)

cap = VideoCap(720, 1080, '')

obj = objDetector('windowmodel.tflite', 720, 1080, conf_thre=0.6)
move = movementDetector(cap.read().copy())
pose = poseDetector()

warn = warning_gen(iou_thre=0.5)
pose_thre = 0.5

counter = Counter()

def obj_f(rq: queue.Queue, dq: queue.Queue) -> None:
    "    窗户目标检测的函数\n\n    内部有sleep()控制运行频率\n\n    该函数为多线程准备\n\n    参数\n    ----------\n    rq : 帧数据队列\n\n    dq : 计算数据队列\n\n    信息交互\n    ----------\n    从 rq 中获取帧\n    将计算数据存入 dq\n\n    当从 rq 中读到 None 时才退出\n"
    while True:
        data = rq.get(block=True)
        if data is None:
            break
        pdt_l = obj.detect(data)
        dq.put(dp(pdt_l, DETECT_TYPE_OBJ), block=True)
        time.sleep(10)


def move_f(rq: queue.Queue, dq: queue.Queue) -> None:
    "    判断运动物体的函数\n\n    该函数为多线程准备\n\n    参数\n    ----------\n    rq : 帧数据队列\n\n    dq : 计算数据队列\n\n    信息交互\n    ----------\n    从 rq 中获取帧\n    将计算数据存入 dq\n\n    当从 rq 中读到 None 时才退出\n\n    性能\n    ----------\n    单独运行时FPS≈10\n"
    while True:
        data = rq.get(block=True)
        if data is None:
            break
        counter.start('move')
        framediff, thresh, diffbox = move.detect(data)
        dq.put(dp((diffbox, thresh), DETECT_TYPE_DIFF), block=True)
        print(counter.end('move'))
        # time.sleep(0.1)


def pose_f(f):
    "    计算姿态的函数\n\n    参数\n    ----------\n    f : BGR格式的帧\n\n    返回\n    ----------\n    tuple(攀爬姿态的几率, landmarks, connection)\n"
    # while True:
    #data = rq.get(block=True)
    if f is None:
        # break
        return
    landmarks, connection = pose.detect(f)
    print('pose detect')
    p = pose.pose_invoke(landmarks)
    #dq.put(dp((landmarks, connection),DETECT_TYPE_POSE),block=True)
    return p, landmarks, connection
    # time.sleep(1)


t_d1 = threading.Thread(target=obj_f, args=(rq, dq))
t_d2 = threading.Thread(target=move_f, args=(rq, dq))

# t_d1.start()
t_d2.start()

d = [None, None, None, None]

while True:
    counter.start('main')
    f = cap.read().copy()
    key = cv2.waitKey(1)
    thresh = None
    isdgrs = False

    if not f.any() or key == ord('q'):
        # 没读到帧或按下q时退出循环
        # 向帧队列放入None 提示子线程退出
        for i in range(2):
            rq.put(None, block=True)
        cap.release()
        cv2.destroyAllWindows()
        break

    if not rq.full():
        rq.put(f, block=True)

    if not dq.empty():
        # 一次读帧只获取一次计算数据
        # 因为读帧速度远大于计算速度，因此不会导致数据队列堵塞
        package = dq.get(block=True)
        TYPE = package.type
        # 按类型更新计算数据
        if TYPE == DETECT_TYPE_OBJ:
            # e[0]:box[ymin,xmin,ymax,xmax]
            # e[1]:class
            # e[2]:score
            d[0] = package.load
            load = np.array(d[0])
            if np.size(load) != 0:
                warn.update_window(
                    load[load[:, 1] == OBJ_IDX_OPENWINDOW][:, 0])
                # 更新窗户或运动物体时进行判断
                isdgrs = warn.close_window()
            else:
                warn.update_window(None)

        elif TYPE == DETECT_TYPE_DIFF:
            d[1] = package.load[0]
            thresh = package.load[1]
            if np.size(d[1]) != 0:
                warn.update_move(d[1])
            else:
                warn.update_move(None)
            # 更新窗户或运动物体时进行判断
            isdgrs = warn.close_window()
        # else:
        #    d[2] = package.load[0]
        #    d[3] = package.load[1]
    if(isdgrs):
        # 运动物体靠近打开窗户时才进行姿态检测
        print('danger start pose')
        prob, lm, conn = pose_f(f)
        d[2] = lm
        d[3] = conn
        print(prob)
        if prob is not None and prob[0] > pose_thre:
            print('danger pose')
    else:
        #print('no danger')
        d[2] = None
        d[3] = None

    if(warn.is_disapr()):
        print('child disapr')

    if(d[0]):
        cap.draw_bbox(d[0])
    if(d[1]):
        cap.draw_diffbox(d[1])
    if(d[2]):
        cap.draw_pose(d[2], d[3])

    cv2.imshow('frame', cap.frame)

    # if not thresh.all():
    # cap.write()
    #cv2.imshow('thresh', thresh)

    if key == ord('r'):
        # 按下r时重置运动检测的基准帧
        move.reset_frame(rq.get(block=True))

    print(counter.end('main'))

# t_d1.join()
t_d2.join()

print('quit')
