
from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import time
import glob
import random
import math
import re
import sys

import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFile


def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    # distortion
    # new_cam[1][0][1] = cam[1][0][1] * scale
    # new_cam[1][1][0] = cam[1][1][0] * scale
    return new_cam


def scale_mvs_camera(cams, scale=1):
    """ resize input in order to produce sampled depth map """
    for view in range(len(cams)):
        cams[view] = scale_camera(cams[view], scale=scale)
    return cams


def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'biculic':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def scale_input(image, cam, depth_image=None, scale=1):
    """ resize input to fit into the memory """
    image = scale_image(image, scale=scale)
    cam = scale_camera(cam, scale=scale)
    if depth_image is None:
        return image, cam
    else:
        depth_image = scale_image(depth_image, scale=scale, interpolation='linear')
        return image, cam, depth_image


def crop_input(image, cam, depth_image=None, max_h=384, max_w=768, resize_scale=1, base_image_size=32):
    """ resize images and cameras to fit the network (can be divided by base image size) """
    # crop images and cameras
    max_h = int(max_h * resize_scale)
    max_w = int(max_w * resize_scale)
    h, w = image.shape[0:2]
    new_h = h
    new_w = w
    if new_h > max_h:
        new_h = max_h
    else:
        new_h = int(math.ceil(h / base_image_size) * base_image_size)
    if new_w > max_w:
        new_w = max_w
    else:
        new_w = int(math.ceil(w / base_image_size) * base_image_size)
    start_h = int(math.ceil((h - new_h) / 2))
    start_w = int(math.ceil((w - new_w) / 2))
    finish_h = start_h + new_h
    finish_w = start_w + new_w
    image = image[start_h:finish_h, start_w:finish_w]
    cam[1][0][2] = cam[1][0][2] - start_w
    cam[1][1][2] = cam[1][1][2] - start_h

    # crop depth image
    if depth_image is not None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return image, cam, depth_image
    else:
        return image, cam


def center_image(img, mode='mean'):
    """ normalize image input """
    # attention: CasMVSNet [mean var];; CasREDNet [0-255]
    if mode == 'standard':
        np_img = np.array(img, dtype=np.float32) / 255.

    elif mode == 'mean':
        img_array = np.array(img)
        img = img_array.astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        np_img = (img - mean) / (np.sqrt(var) + 0.00000001)

    elif mode == 'vit':
        img_array = np.array(img)
        img = img_array.astype(np.float32)
        pixel_mean = np.array([123.675, 116.28, 103.53]).astype(np.float32)
        pixel_std = np.array([58.395, 57.12, 57.375]).astype(np.float32)
        np_img = (img - pixel_mean) / (pixel_std + 0.00000001)

    else:
        raise Exception("{}? Not implemented yet!".format(mode))

    return np_img



# data augment
def image_augment(image):

    image = randomColor(image)
    #image = randomGaussian(image, mean=0.2, sigma=0.3)

    return image
    

def randomColor(image):

    random_factor = np.random.randint(1, 301) / 100.
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # Image Color
    random_factor = np.random.randint(10, 201) / 100.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # Image Brightness
    random_factor = np.random.randint(10, 201) / 100.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # Image Contrast
    random_factor = np.random.randint(0, 301) / 100.
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # Image Sharpness

    return sharpness_image


def randomGaussian(image, mean=0.02, sigma=0.03):

    def gaussianNoisy(im, mean=0.02, sigma=0.03):


        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    img.flags.writeable = True
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])

    return Image.fromarray(np.uint8(img))


class RandomCrop(object):
    def __init__(self, CropSize=0.1):
        self.CropSize = CropSize

    def __call__(self, image, normal):
        import random, cv2
        h, w = normal.shape[:2]
        img_h, img_w = image.shape[:2]
        CropSize_w, CropSize_h = max(1, int(w * self.CropSize)), max(1, int(h * self.CropSize))
        x1, y1 = random.randint(0, CropSize_w), random.randint(0, CropSize_h)
        x2, y2 = random.randint(w - CropSize_w, w), random.randint(h - CropSize_h, h)

        normal_crop = normal[y1:y2, x1:x2]
        normal_resize = cv2.resize(normal_crop, (w, h), interpolation=cv2.INTER_NEAREST)

        image_crop = image[4*y1:4*y2, 4*x1:4*x2]
        image_resize = cv2.resize(image_crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        return image_resize, normal_resize