import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from PIL import Image
from torch import nn


class Parking:

    def show_images(self, images, cmap=None):
        cols = 2
        rows = (len(images) + 1) // cols

        plt.figure(figsize=(15, 12))
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            cmap = 'gray' if len(image.shape) == 2 else cmap
            plt.imshow(image, cmap=cmap)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.show()

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 过滤掉背景
    def select_rgb_white_yellow(self, image):

        lower = np.uint8([120, 120, 120])
        upper = np.uint8([255, 255, 255])
        # lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255,相当于过滤背景
        white_mask = cv2.inRange(image, lower, upper)
        self.cv_show('white_mask', white_mask)

        # 与原图像做与操作，实现对原图像的过滤
        masked = cv2.bitwise_and(image, image, mask=white_mask)
        self.cv_show('masked', masked)
        return masked

    # 转换灰度
    def convert_gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 边缘检测
    def detect_edges(self, image, low_threshold=50, high_threshold=200):
        return cv2.Canny(image, low_threshold, high_threshold)

    # 过滤无效区域
    def filter_region(self, image, vertices):
        """
                剔除掉不需要的地方
        """
        # 纯黑底图
        mask = np.zeros_like(image)

        if len(mask.shape) == 2:
            # 构建mask，对有效区域内填充白色
            cv2.fillPoly(mask, vertices, 255)
            self.cv_show('mask', mask)
        # 与操作，进行过滤
        return cv2.bitwise_and(image, mask)

    # 手动选择有效区域
    def select_region(self, image):
        """
                手动选择区域
        """
        rows, cols = image.shape[:2]
        # 手动规定临界点
        pt_1 = [cols * 0.05, rows * 0.90]
        pt_2 = [cols * 0.05, rows * 0.70]
        pt_3 = [cols * 0.30, rows * 0.55]
        pt_4 = [cols * 0.6, rows * 0.15]
        pt_5 = [cols * 0.90, rows * 0.15]
        pt_6 = [cols * 0.90, rows * 0.90]

        vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
        point_img = image.copy()
        point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)

        # 画出临界点
        for point in vertices[0]:
            cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
        self.cv_show('point_img', point_img)

        return self.filter_region(image, vertices)

    # 直线检测
    def hough_lines(self, image):
        # 输入的图像需要是边缘检测后的结果
        # minLineLengh(线的最短长度，比这个短的都被忽略)和MaxLineCap（两条直线之间的最大间隔，小于此值，认为是一条直线）
        # rho距离精度,theta角度精度,threshod超过设定阈值才被检测出线段
        return cv2.HoughLinesP(image, rho=0.1, theta=np.pi / 10, threshold=15, minLineLength=9, maxLineGap=4)

    # 绘制直线
    def draw_lines(self, image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
        # 过滤霍夫变换检测到直线
        if make_copy:
            image = np.copy(image)
        cleaned = []
        # 过滤
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 只保留横向直线，长度在25到55之间
                if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))
                    # 画线
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        print(" No lines detected: ", len(cleaned))
        return image

    def identify_blocks(self, image, lines, make_copy=True):
        if make_copy:
            new_image = np.copy(image)
        # Step 1: 过滤部分直线
        cleaned = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))

        # Step 2: 对直线按照x1进行排序，分出列顺序
        import operator
        list1 = sorted(cleaned, key=operator.itemgetter(0, 1))

        # Step 3: 找到多个列，相当于每列是一排车
        clusters = {}
        dIndex = 0
        # 最大列内距离，同一列中直线x1的最大距离
        clus_dist = 10

        for i in range(len(list1) - 1):
            # 计算距离
            distance = abs(list1[i + 1][0] - list1[i][0])

            if distance <= clus_dist:
                # 同一列中的直线
                # 初始化列
                if not dIndex in clusters.keys():
                    clusters[dIndex] = []
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i + 1])

            else:
                # 不同列中直线，列数加1
                dIndex += 1

        # Step 4: 得到坐标
        rects = {}
        i = 0
        for key in clusters:
            # 取出列中的所有线
            all_list = clusters[key]
            # 去重
            cleaned = list(set(all_list))
            # 超过五条线视为一列
            if len(cleaned) > 5:
                # 按照y大小上下排序
                cleaned = sorted(cleaned, key=lambda tup: tup[1])
                # 最低线y值
                avg_y1 = cleaned[0][1]
                # 最高线y值
                avg_y2 = cleaned[-1][1]
                avg_x1 = 0
                avg_x2 = 0
                # 求平均x1和x2，即整列的左右两边
                for tup in cleaned:
                    avg_x1 += tup[0]
                    avg_x2 += tup[2]
                avg_x1 = avg_x1 / len(cleaned)
                avg_x2 = avg_x2 / len(cleaned)
                # 整列的矩形框
                rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
                i += 1

        print("Num Parking Lanes: ", len(rects))
        # Step 5: 把列矩形画出来
        #   将列范围扩大一些
        buff = 7
        for key in rects:
            tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
            tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
            cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 3)
        return new_image, rects

    # 划停车位
    def draw_parking(self, image, rects, make_copy=True, color=[255, 0, 0], thickness=2, save=True):
        if make_copy:
            new_image = np.copy(image)
        # 固定车位宽度
        gap = 15.5
        spot_dict = {}  # 字典：一个车位对应一个位置
        tot_spots = 0
        # 对识别的车位信息进行手工微调
        adj_y1 = {0: 20, 1: -10, 2: 0, 3: -11, 4: 28, 5: 5, 6: -15, 7: -15, 8: -10, 9: -30, 10: 9, 11: -32}
        adj_y2 = {0: 30, 1: 50, 2: 15, 3: 10, 4: -15, 5: 15, 6: 15, 7: -20, 8: 15, 9: 15, 10: 0, 11: 30}

        adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10, 11: 0}
        adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}
        for key in rects:
            tup = rects[key]
            x1 = int(tup[0] + adj_x1[key])
            x2 = int(tup[2] + adj_x2[key])
            y1 = int(tup[1] + adj_y1[key])
            y2 = int(tup[3] + adj_y2[key])

            # 画出微调后的矩形框
            cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 计算列内车位
            num_splits = int(abs(y2 - y1) // gap)

            # 计算每一停车位
            for i in range(0, num_splits + 1):
                # 画出标准车位分隔线
                y = int(y1 + i * gap)
                cv2.line(new_image, (x1, y), (x2, y), color, thickness)
            # 除了第一列和最后一列外，其余列为双车位，需要画竖线
            if key > 0 and key < len(rects) - 1:
                # 从中间竖直线
                x = int((x1 + x2) / 2)
                cv2.line(new_image, (x, y1), (x, y2), color, thickness)
            # 计算数量
            if key == 0 or key == (len(rects) - 1):
                # 单车位
                tot_spots += num_splits + 1
            else:
                # 双车位
                tot_spots += 2 * (num_splits + 1)

            # 字典对应好
            if key == 0 or key == (len(rects) - 1):
                # 单车位列
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    spot_dict[(x1, y, x2, y + gap)] = cur_len + 1
            else:
                # 双车位列，一次录入两个车位
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    x = int((x1 + x2) / 2)
                    spot_dict[(x1, y, x, y + gap)] = cur_len + 1
                    spot_dict[(x, y, x2, y + gap)] = cur_len + 2

        print("total parking spaces: ", tot_spots, cur_len)
        if save:
            filename = 'with_parking.jpg'
            cv2.imwrite(filename, new_image)
        return new_image, spot_dict

    def assign_spots_map(self, image, spot_dict, make_copy=True, color=[255, 0, 0], thickness=2):
        if make_copy:
            new_image = np.copy(image)
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            cv2.rectangle(new_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return new_image

    # 裁剪车位图片
    def save_images_for_cnn(self, image, spot_dict, folder_name='cnn_data'):
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            # 裁剪
            spot_img = image[y1:y2, x1:x2]
            spot_img = cv2.resize(spot_img, (0, 0), fx=2.0, fy=2.0)
            spot_id = spot_dict[spot]

            filename = 'spot' + str(spot_id) + '.jpg'
            print(spot_img.shape, filename, (x1, x2, y1, y2))

            cv2.imwrite(os.path.join(folder_name, filename), spot_img)

    # 神经网络预测
    def make_prediction(self, image, net, class_dictionary, device):

        # 预变换
        test_augs = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
        ])
        # np.array -> PIL image
        image = Image.fromarray(image)
        # 变换
        image = test_augs(image)
        # 添加批量维度
        image = torch.unsqueeze(image, 0)
        # 移至GPU
        image = image.to(device)
        # 统计预测值
        inID = net(image).argmax(axis=1)
        # 提取tensor中的数据
        inID = inID.item()
        # 返回标签
        label = class_dictionary[inID]
        return label

    # 神经网络识别
    def predict_on_image(self, image, spot_dict, model, class_dictionary, device, make_copy=True, color=[0, 255, 0],
                         alpha=0.5):
        if make_copy:
            new_image = np.copy(image)
            overlay = np.copy(image)
        cnt_empty = 0
        all_spots = 0
        for spot in spot_dict.keys():
            all_spots += 1
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            # 裁切图像
            spot_img = image[y1:y2, x1:x2]

            # 神经网络预测
            label = self.make_prediction(spot_img, model, class_dictionary, device)

            # 标记空车位位置
            if label == 'empty':
                # 实心矩形
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                cnt_empty += 1

        # 画图
        # 叠加绿色车位图
        cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

        cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        save = False

        if save:
            filename = 'with_marking.jpg'
            cv2.imwrite(filename, new_image)

        return new_image

    def predict_on_video(self, video_name, final_spot_dict, model, class_dictionary, device, ret=True):
        cap = cv2.VideoCapture(video_name)
        count = 0
        while ret:
            ret, image = cap.read()
            count += 1
            # 每5帧检测一次
            if count == 10:
                count = 0

                new_image = np.copy(image)
                overlay = np.copy(image)
                cnt_empty = 0
                all_spots = 0
                color = [0, 255, 0]
                alpha = 0.5

                for spot in final_spot_dict.keys():
                    all_spots += 1
                    (x1, y1, x2, y2) = spot
                    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                    # 裁剪图像
                    spot_img = image[y1:y2, x1:x2]
                    spot_img = cv2.resize(spot_img, (48, 48))
                    # 预测
                    label = self.make_prediction(spot_img, model, class_dictionary, device)
                    if label == 'empty':
                        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                        cnt_empty += 1

                cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

                cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

                cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.imshow('frame', new_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        cap.release()
