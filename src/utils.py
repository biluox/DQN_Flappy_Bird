import cv2
import numpy as np
import torch


def resize_and_bgr2gray(image):
    # 去除地面图像
    image = image[:288, :404]  # 512*0.79
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    # 二值化
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def image_to_tensor(image):
    # 原shape(84,84,1) torch默认通道数在最前面
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    return image_tensor
