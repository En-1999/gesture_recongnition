import win32api
import win32con
import cv2 as cv
import numpy as np
import copy
import math


# 形态学开操作
def open_binary(binary, x, y):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # 获取图像结构化元素
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)  # 开操作
    return dst


# 形态学闭操作
def close_binary(binary, x, y):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # 获取图像结构化元素
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)  # 开操作
    return dst


# 形态学腐蚀操作
def erode_binary(binary, x, y):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # 获取图像结构化元素
    dst = cv.erode(binary, kernel)  # 腐蚀
    return dst


# 形态学膨胀操作
def dilate_binary(binary, x, y):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # 获取图像结构化元素
    dst = cv.dilate(binary, kernel)  # 膨胀返回
    return dst


def body_detetc(frame):

    # ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)  # 分解为YUV图像,得到CR分量
    # (_, cr, _) = cv.split(ycrcb)
    # cr1 = cv.GaussianBlur(cr, (3, 3), 0)  # 高斯滤波
    # _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # OTSU图像二值化
    # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # hsv 色彩空间 分割肤色
    ycrcb = cv.cvtColor(frame,  cv.COLOR_BGR2YCrCb)  # Ycrcb 色彩空间 分割肤色
    # lower_hsv = np.array([0, 15, 0])
    # upper_hsv = np.array([17, 170, 255])
    lower_ycrcb = np.array([0, 135, 85])
    upper_ycrcb = np.array([255, 180, 135])
    # mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)  # hsv 掩码
    mask = cv.inRange(ycrcb, lowerb=lower_ycrcb, upperb=upper_ycrcb)  # ycrcb 掩码
    return mask


def get_roi(frame, x1, x2, y1, y2):
    dst = frame[y1:y2, x1:x2]
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    return dst


if __name__ == "__main__":

    capture = cv.VideoCapture(0)
    m_0 = 1500
    m_1 = 1500
    m_2 = 1500
    m_3 = 1500
    m_4 = 1500
    m_5 = 1500
    m_6 = 1500
    m_7 = 1500
    m_8 = 1500
    m_9 = 1500

    while True:

        ret, frame = capture.read()
        roi = get_roi(frame, 100, 250, 100, 250)
        # skin = body_detetc(roi)
        # close = close_binary(skin, 8, 8)
        # dilate = dilate_binary(skin, 5, 5)
        # erode = erode_binary(dilate, 5, 5)
        # cv.imshow("close = ", close)

        k = cv.waitKey(50)
        if k == 27:  # 按下ESC退出
            break
        elif k == ord('a'):  # 按下'b'会捕获背景
            cv.imwrite("E:\\aiFile\\picture\\gesture_data\\0\\%s.jpg" % m_0, roi)
            m_0 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\0\\%s.jpg" % m_0, flip_image)
            # m_0 += 1
            print('正在保存0-roi图片,本次图片数量:', m_0)
        elif k == ord('s'):  # 按下'b'会捕获背景
            cv.imwrite("E:\\aiFile\\picture\\gesture_data\\1\\%s.jpg" % m_1, roi)
            m_1 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\1\\%s.jpg" % m_1, flip_image)
            # m_1 += 1
            print('正在保存1-roi图片,本次图片数量:', m_1)
        elif k == ord('d'):  # 按下'b'会捕获背景
            cv.imwrite("E:\\aiFile\\picture\\gesture_data\\2\\%s.jpg" % m_2, roi)
            m_2 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\2\\%s.jpg" % m_2, flip_image)
            # m_2 += 1
            print('正在保存2-roi图片,本次图片数量:', m_2)
        elif k == ord('f'):  # 按下'b'会捕获背景
            cv.imwrite("E:\\aiFile\\picture\\gesture_data\\3\\%s.jpg" % m_3, roi)
            m_3 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\3\\%s.jpg" % m_3, flip_image)
            # m_3 += 1
            print('正在保存3-roi图片,本次图片数量:', m_3)
        elif k == ord('g'):  # 按下'b'会捕获背景
            cv.imwrite("E:\\aiFile\\picture\\gesture_data\\4\\%s.jpg" % m_4, roi)
            m_4 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\4\\%s.jpg" % m_4, flip_image)
            # m_4 += 1
            print('正在保存4-roi图片,本次图片数量:', m_4)
        elif k == ord('h'):  # 按下'b'会捕获背景
            cv.imwrite("E:\\aiFile\\picture\\gesture_data\\5\\%s.jpg" % m_5, roi)
            m_5 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\5\\%s.jpg" % m_5, flip_image)
            # m_5 += 1
            print('正在保存5-roi图片,本次图片数量:', m_5)
        elif k == ord('j'):  # 按下'b'会捕获背景
            cv.imwrite("E:\\aiFile\\picture\\gesture_data\\6\\%s.jpg" % m_6, roi)
            m_6 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\6\\%s.jpg" % m_6, flip_image)
            # m_6 += 1
            print('正在保存6-roi图片,本次图片数量:', m_6)
        elif k == ord('k'):  # 按下'b'会捕获背景
            cv.imwrite("E:\\aiFile\\picture\\gesture_data\\7\\%s.jpg" % m_7, roi)
            m_7 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\7\\%s.jpg" % m_7, flip_image)
            # m_7 += 1
            print('正在保存7-roi图片,本次图片数量:', m_7)
        elif k == ord('l'):  # 按下'b'会捕获背景
            cv.imwrite("E:\\aiFile\\picture\\gesture_data\\8\\%s.jpg" % m_8, roi)
            m_8 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\8\\%s.jpg" % m_8, flip_image)
            # m_8 += 1
            print('正在保存8-roi图片,本次图片数量:', m_8)
        elif k == ord('z'):  # 按下'b'会捕获背景
            cv.imwrite("E:\\aiFile\\picture\\gesture_data\\9\\%s.jpg" % m_9, roi)
            m_9 += 1
            # flip_image = cv.flip(skin, 1)  # 这里用到的是水平翻转，因为后面的参数是一
            # cv.imwrite("E:\\aiFile\\picture\\gesture_data\\9\\%s.jpg" % m_9, flip_image)
            # m_9 += 1
            print('正在保存9-roi图片,本次图片数量:', m_9)
        cv.imshow("roi", roi)
        cv.imshow("frame", frame)
        c = cv.waitKey(50)
        if c == 27:
            break
    cv.waitKey(0)
    capture.release()
    cv.destroyAllWindows()
