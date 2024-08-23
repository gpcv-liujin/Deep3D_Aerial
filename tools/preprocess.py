#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright (c) 2024, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu


import os
import sys
sys.path.append("..")
import cv2
import numpy as np
import random
import math
import shutil

from IO.params_io import read_images_path_text, read_predef_cameras_text, read_predef_images_text, read_center_offset_text
from tools.utils import join, mk_father_dir_if_not_exist


"""
data preprocessing.
"""


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

    else:
        raise Exception("{}? Not implemented yet!".format(mode))

    return np_img


def scale_cam(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # size
    new_cam[0] = cam[0] * scale
    new_cam[1] = cam[1] * scale
    # focal:
    new_cam[2] = cam[2] * scale
    new_cam[3] = cam[3] * scale
    # principle
    new_cam[4] = cam[4] * scale
    new_cam[5] = cam[5] * scale
    # distortion
    # new_cam[1] = cam[1] * scale
    # new_cam[0] = cam[0] * scale
    return new_cam

def crop_cam(cam, max_h=384, max_w=768):
    """ resize images and cameras to fit the network (can be divided by base image size) """
    new_cam = np.copy(cam)
    # crop cameras
    h, w = cam[1], cam[0]
    new_h = h
    new_w = w

    if new_h > max_h:
        new_h = max_h
    if new_w > max_w:
        new_w = max_w

    start_h = int(math.ceil((h - new_h) / 2))
    start_w = int(math.ceil((w - new_w) / 2))


    new_cam[0] = new_w
    new_cam[1] = new_h
    new_cam[4] = cam[4] - start_w
    new_cam[5] = cam[5] - start_h

    return new_cam


def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'biculic':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def crop_image(image, max_h=384, max_w=768):
    """ resize images and cameras to fit the network (can be divided by base image size) """
    # crop images and cameras
    h, w = image.shape[0:2]
    new_h = h
    new_w = w

    if new_h > max_h:
        new_h = max_h
    if new_w > max_w:
        new_w = max_w

    start_h = int(math.ceil((h - new_h) / 2))
    start_w = int(math.ceil((w - new_w) / 2))
    finish_h = start_h + new_h
    finish_w = start_w + new_w
    image = image[start_h:finish_h, start_w:finish_w]

    return image


def crop_and_scale_images(orig_image_path_txt, save_image_path, image_h, image_w, image_scale):
    photo_paths = []
    paths_list, names_list = read_images_path_text(orig_image_path_txt)

    for index in paths_list.keys():
        name = names_list[index]
        path = paths_list[index]

        new_path = join(save_image_path, name + '.jpg')
        mk_father_dir_if_not_exist(new_path)

        print("------>crop and scale image {}".format(name + '.jpg'))
        src = cv2.imread(path)
        crop_src = crop_image(src, max_h=image_h, max_w=image_w)
        scale_src = scale_image(crop_src, scale=image_scale, interpolation='biculic')
        cv2.imwrite(new_path, scale_src)

        photo_paths.append((index, name, new_path))

    return photo_paths


def crop_and_scale_params(orig_cam_txt_path, curr_cam_txt_path,  orig_image_txt_path, curr_image_txt_path, orig_offset_txt, curr_offset_txt,
                          photo_paths_list, curr_image_path_txt, image_h, image_w, image_scale):

    if photo_paths_list:
        f = open(curr_image_path_txt, "w")
        f.write('%d\n' % len(photo_paths_list))
        for index, image_name, image_path in photo_paths_list:
            f.write('%d ' % index)
            f.write('%s ' % image_name)
            f.write('%s\n' % image_path)
        f.close()

    # write images and cams for this project
    cams = read_predef_cameras_text(orig_cam_txt_path)
    images = read_predef_images_text(orig_image_txt_path)
    if os.path.exists(orig_offset_txt):
        offset = read_center_offset_text(orig_offset_txt)
    else:
        offset = [0.0, 0.0, 0.0]

    f = open(curr_cam_txt_path, "w")
    f.write('# Number of cameras: %d\n' % len(cams))
    f.write('# CAMERA_MODEL: OPENCV\n')
    f.write('# Camera list with one line of data per camera:\n')
    f.write('# CAMERA_ID, WIDTH, HEIGHT, PIXELSIZE, PARAMS[fx,fy,cx,cy], DISTORTION[K1, K2, P1, P2]\n')

    for id in cams.keys():
        cam = cams[id]
        id = cam.camera_id
        width = cam.size[0]
        height = cam.size[1]
        pixel = cam.pixelsize
        focallength_x = cam.focallength[0]
        focallength_y = cam.focallength[1]
        x0 = cam.x0y0[0]
        y0 = cam.x0y0[1]
        distortion = cam.distortion
        curr_src = [width, height, focallength_x, focallength_y, x0, y0]
        crop_src = crop_cam(curr_src, max_h=image_h, max_w=image_w)
        scale_src = scale_cam(crop_src, scale=image_scale)

        f.write('%d %d %d ' % (id, scale_src[0], scale_src[1]))  ## CAMERA_ID, WIDTH, HEIGHT
        f.write('%.6f %.6f %.6f %.6f %.6f ' % (pixel, scale_src[2], scale_src[3], scale_src[4],
                                               scale_src[5]))  ## PIXELSIZE, PARAMS[fx,fy,cx,cy]
        f.write('%.6f %.6f %.6f %.6f\n' % (
            distortion[0], distortion[1], distortion[2], distortion[3]))  ## DISTORTION[K1, K2, P1, P2]

    f.close()

    # write images_info.txt
    f = open(curr_image_txt_path, "w")
    f.write('# Number of images: %d\n' % len(images))
    f.write("# Image list with two lines of data per image:\n")
    f.write("# CAMERA ORI: [ XrightYup | Rwc | twc ]\n")
    f.write('#  IMAGE_ID, CAMERA_ID, Rwc[9], twc[3], MINDEPTH, MAXDEPTH, NAME\n')

    for id in images.keys():
        photo = images[id]
        orig_name = photo.name
        new_name = os.path.splitext(orig_name)[0] + '.jpg'
        f.write('%d %d ' % (photo.image_id, photo.camera_id))
        f.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f ' % (
        photo.rotation_matrix[0, 0], photo.rotation_matrix[0, 1], photo.rotation_matrix[0, 2],
        photo.rotation_matrix[1, 0], photo.rotation_matrix[1, 1], photo.rotation_matrix[1, 2],
        photo.rotation_matrix[2, 0], photo.rotation_matrix[2, 1], photo.rotation_matrix[2, 2]))
        f.write('%.6f %.6f %.6f ' % (photo.project_center[0]-offset[0], photo.project_center[1]-offset[1], photo.project_center[2]-offset[2]))
        f.write('%.6f %.6f ' % (photo.depth[0], photo.depth[1]))
        f.write('%s\n' % (new_name))
    f.close()

    f = open(curr_offset_txt, "w")
    f.write('# Center offset\n')
    f.write('# X_offset  |  Y_offset  | Z_offset\n')
    f.write('%f\n' % offset[0])
    f.write('%f\n' % offset[1])
    f.write('%f\n' % offset[2])
    f.close()



















