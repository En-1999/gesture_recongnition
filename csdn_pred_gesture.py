import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import pathlib
import random
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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


def nothing(x):
    pass


def creatTrackbar():

    cv.createTrackbar("x1", "roi_adjust", 200, 800, nothing)
    cv.createTrackbar("x2", "roi_adjust", 400, 800, nothing)
    cv.createTrackbar("y1", "roi_adjust", 100, 800, nothing)
    cv.createTrackbar("y2", "roi_adjust", 300, 800, nothing)


def get_roi(frame, x1, x2, y1, y2):
    dst = frame[y1:y2, x1:x2]
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    return dst


def body_detetc(frame):

    # ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)  # 分解为YUV图像,得到CR分量
    # (_, cr, _) = cv.split(ycrcb)
    # cr1 = cv.GaussianBlur(cr, (3, 3), 0)  # 高斯滤波
    # _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # OTSU图像二值化
    # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # hsv 色彩空间 分割肤色
    ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)  # Ycrcb 色彩空间 分割肤色
    # lower_hsv = np.array([0, 15, 0])
    # upper_hsv = np.array([17, 170, 255])
    lower_ycrcb = np.array([0, 135, 85])
    upper_ycrcb = np.array([255, 180, 135])
    # mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)  # hsv 掩码
    mask = cv.inRange(ycrcb, lowerb=lower_ycrcb, upperb=upper_ycrcb)  # ycrcb 掩码
    return mask


def get_image(image, network):

    # image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.expand_dims(image, axis=2)  # 扩维度,让单通道图片可以resize
    print("image.shape =  ", image.shape)
    image = tf.image.resize(image, [100, 100])
    # image = cv.resize(image, (100, 100))
    # image = image.reshape(-1, 100, 100,  1)
    image1 = image / 255.0  # normalize to [0,1] range
    image1 = tf.expand_dims(image1, axis=0)
    # print(image1.shape)
    pred = network(image1)
    print("预测结果原始结果", pred)
    print()
    pred = tf.nn.softmax(pred['output_1'], axis=1)
    print("预测softmax后", pred)
    pred = tf.argmax(pred, axis=1)
    print("最终测试结果", pred.numpy())
    cv.putText(frame, "Predicted results = :" + str(pred.numpy()), (100, 400), cv.FONT_HERSHEY_SIMPLEX,
               1, [0, 255, 255])

if __name__ == "__main__":

    capture = cv.VideoCapture(0)
    creatTrackbar()
    channels = 3
    DEFAULT_FUNCTION_KEY = "serving_default"
    loaded = tf.saved_model.load('E:\\aiFile\\model_save\\gesture_recognition_model\\gestureModel_one\\')
    network = loaded.signatures[DEFAULT_FUNCTION_KEY]
    print(list(loaded.signatures.keys()))
    print('加载 weights 成功')
    while True:

        dx1 = cv.getTrackbarPos('x1', 'roi_adjust')
        dx2 = cv.getTrackbarPos('x2', 'roi_adjust')
        dy1 = cv.getTrackbarPos('y1', 'roi_adjust')
        dy2 = cv.getTrackbarPos('y2', 'roi_adjust')
        ret, frame = capture.read()
        roi = get_roi(frame, 100, 250, 100, 250)
        # skin = body_detetc(roi)
        # dilate = dilate_binary(skin, 5, 5)
        # erode = erode_binary(dilate, 5, 5)
        # print("skin.shape", skin.shape)
        # print("x1, x2, y1, y2 ", dx1, dx2, dy1, dy2)
        cv.imshow("roi", roi)
        get_image(roi, network)
        cv.imshow("frame", frame)
        c = cv.waitKey(50)
        if c == 27:
            break
    cv.waitKey(0)
    capture.release()
    cv.destroyAllWindows()
    """图片预测
    # path = 'E:\\aiFile\\picture\\gesture_data\\5\\45.jpg'
    # test_image = plt.imread(path)
    # image = tf.io.read_file(path)
    # image = tf.image.decode_jpeg(image, channels=channels)
    # image = tf.image.resize(image, [100, 100])
    # image1 = image / 255.0  # normalize to [0,1] range
    # image1 = tf.expand_dims(image1, axis=0)
    # # print(image1.shape)
    # pred = network(image1)
    # print("预测结果原始结果", pred)
    # print()
    # pred = tf.nn.softmax(pred['output_1'], axis=1)
    # print("预测softmax后", pred)
    # pred = tf.argmax(pred, axis=1)
    # print("最终测试结果", pred.numpy())
    """
