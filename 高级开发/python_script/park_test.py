from __future__ import division
import cv2
import os, glob

from d2l.torch import d2l
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from Parking import Parking
import pickle
from PIL import Image
import numpy as np

cwd = os.getcwd()

park = Parking()


def img_process(test_images, park, modelName):
    # 背景过滤
    white_yellow_images = list(map(park.select_rgb_white_yellow, test_images))
    park.show_images(white_yellow_images)

    # 转换为灰度图
    gray_images = list(map(park.convert_gray_scale, white_yellow_images))
    park.show_images(gray_images)

    # 边缘检测
    edge_images = list(map(lambda image: park.detect_edges(image), gray_images))
    park.show_images(edge_images)

    # 选择停车场区域
    roi_images = list(map(park.select_region, edge_images))
    park.show_images(roi_images)

    # 直线检测，检测停车位四周的直线
    list_of_lines = list(map(park.hough_lines, roi_images))

    # 画线
    line_images = []
    for image, lines in zip(test_images, list_of_lines):
        line_images.append(park.draw_lines(image, lines))
    park.show_images(line_images)

    # 划分停车位的列
    rect_images = []
    rect_coords = []
    for image, lines in zip(test_images, list_of_lines):
        # 识别区域
        new_image, rects = park.identify_blocks(image, lines)
        # 图片与列矩形信息
        rect_images.append(new_image)
        rect_coords.append(rects)

    park.show_images(rect_images)

    # 对于每一列切分停车位
    delineated = []
    spot_pos = []
    for image, rects in zip(test_images, rect_coords):
        new_image, spot_dict = park.draw_parking(image, rects)
        # 图片与车位矩形信息
        delineated.append(new_image)
        spot_pos.append(spot_dict)

    park.show_images(delineated)

    # 选择图1的结果
    final_spot_dict = spot_pos[0]
    print(len(final_spot_dict))

    # 保存车位坐标
    with open(modelName, 'wb') as handle:
        pickle.dump(final_spot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 裁剪出供训练的图片
    # park.save_images_for_cnn(test_images[0], final_spot_dict)

    return final_spot_dict


def img_test(test_images, final_spot_dict, model, class_dictionary, device):
    predicted_images = []
    for i in range(len(test_images)):
        predicted_images.append(park.predict_on_image(test_images[i], final_spot_dict, model, class_dictionary, device))
    return predicted_images


def video_test(video_name, final_spot_dict, model, class_dictionary, device):
    name = video_name
    cap = cv2.VideoCapture(name)
    park.predict_on_video(name, final_spot_dict, model, class_dictionary, device, ret=True)


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 车位预测函数
def predict(net, model, path, imageName):
    test_images = [np.array(Image.open(path))]
    class_dictionary = {}
    class_dictionary[0] = 'empty'
    class_dictionary[1] = 'occupied'

    # 加载网络模型
    net = torch.load(net)

    # 加载车位信息
    f = open(model, 'rb')
    final_spot_dict = pickle.load(f)

    predicted_images = img_test(test_images, final_spot_dict, net, class_dictionary, d2l.try_gpu())
    print(len(predicted_images))
    for i in predicted_images:
        im = Image.fromarray(i)
        im.save(imageName)
    return imageName


def init_spot(path, modelName):
    test_images = [np.array(Image.open(path))]
    img_process(test_images, park, modelName)
    return 1


if __name__ == '__main__':
    # 预测
    predict('data/net', 'data/model.pickle', 'test_images/scene1380.jpg', 'data/test1.jpg')
    # 初始化
    # init_spot('test_images/scene1380.jpg', 'data/model.pickle')
