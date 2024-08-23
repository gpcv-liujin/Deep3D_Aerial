# Copyright (c) 2022, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu

from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import numpy as np
import re
import sys
sys.path.append("..")
import math
import cv2
import random
import matplotlib.pyplot as plt
from format import cameras


def read_colmap_cameras_text(path):
    cameras = dict()

    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                camera_item = dict()
                elems = line.split()
                camera_id = int(elems[0])
                camera_item['model'] = elems[1]
                camera_item['width'] = int(elems[2])
                camera_item['height'] = int(elems[3])
                camera_item['params'] = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = camera_item

    return cameras


def read_colmap_images_text(path):
    images = dict()

    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                image_item = dict()

                elems = line.split()
                image_id = int(elems[0])
                image_item['qvec'] = np.array(tuple(map(float, elems[1:5])))
                image_item['tvec'] = np.array(tuple(map(float, elems[5:8])))
                image_item['camera_id'] = int(elems[8])
                image_item['image_name'] = elems[9]

                images[image_id] = image_item

    return images


# predef parameters
def read_predef_cameras_text(path):
    """
    OPENCV: CAMERA_ID, WIDTH, HEIGHT, PIXELSIZE, PARAMS[fx,fy,cx,cy], DISTORTION[K1, K2, P1, P2]
    """
    cams = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                width = int(elems[1])
                height = int(elems[2])
                pixelsize = float(elems[3])
                params = np.array(tuple(map(float, elems[4:8])))
                distortion = np.array(tuple(map(float, elems[8:])))
                cams[camera_id] = cameras.Camera(camera_id=camera_id, camera_model='OPENCV', size=[width, height], pixelsize=pixelsize,
                                         focallength=[params[0], params[1]], x0y0=[params[2], params[3]],
                                         distortion=distortion)

    return cams


def read_predef_images_text(path):
    """
    IMAGE_ID, CAMERA_ID, Rwc[9], twc[3], MINDEPTH, MAXDEPTH, NAME
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                camera_id = int(elems[1])
                R_matrix = np.array(tuple(map(float, elems[2:11]))).reshape(3, 3)
                t_matrix = np.array(tuple(map(float, elems[11:14])))
                depth_range = np.array(tuple(map(float, elems[14:16])))
                image_name = elems[16]

                images[image_id] = cameras.Photo(image_id=image_id, camera_id=camera_id, rotation_matrix=R_matrix,
                                         project_center=t_matrix, depth=depth_range, name=image_name)

    return images


def read_center_offset_text(path):
    """
    offset: X Y Z
    """
    offset = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                offset.append(float(elems[0]))

        offset = np.array(offset)
        print(offset)

    return offset


# rednet parameters
def read_rednet_cameras_text(path):
    # read intrinsics and extrinsics
    with open(path) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix  Twc
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4)) 
    # R = extrinsics[0:3,0:3]
    # R2 = np.linalg.inv(R)
    # extrinsics[0:3, 0:3] = R2  # if except for Rwc
    # intrinsics: line [7), 1x3 matrix
    intrinsics = np.fromstring(' '.join(lines[6:7]), dtype=np.float32, sep=' ')
    depthrange = np.fromstring(' '.join(lines[8:9]), dtype=np.float32, sep=' ') 

    return intrinsics, extrinsics, depthrange


def write_rednet_cameras_text(file, cam, location):
    f = open(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    for word in location:
        f.write(str(word) + ' ')
    f.write('\n')

    f.close()


def write_params_for_rednet(file, cam_params, photo_params):
    dir = os.path.dirname(file)
    if not os.path.exists(dir):
        os.mkdir(dir)

    f = open(file, "w")
    f.write('Twc\n')

    rotation_mat = photo_params.rotation_matrix
    project_center = photo_params.project_center

    f.write('%.6f %.6f %.6f %.6f\n' % (rotation_mat[0, 0], rotation_mat[0, 1], rotation_mat[0, 2], project_center[0]))
    f.write('%.6f %.6f %.6f %.6f\n' % (rotation_mat[1, 0], rotation_mat[1, 1], rotation_mat[1, 2], project_center[1]))
    f.write('%.6f %.6f %.6f %.6f\n' % (rotation_mat[2, 0], rotation_mat[2, 1], rotation_mat[2, 2], project_center[2]))
    f.write('%.6f %.6f %.6f %.6f\n' % (0.0, 0.0, 0.0, 1.0))
    f.write('\n')

    f.write('K_mat\n')
    f.write('%.6f %.6f %.6f\n' % (cam_params.focallength[0], 0.0, cam_params.x0y0[0]))
    f.write('%.6f %.6f %.6f\n' % (0.0, cam_params.focallength[1], cam_params.x0y0[1]))
    f.write('%.6f %.6f %.6f\n' % (0.0, 0.0, 1.0))
    f.write('\n')

    f.write('%.6f %.6f %.6f\n' % (photo_params.depth[0], photo_params.depth[1], 0.1))
    f.write('\n')

    Width = cam_params.size[0]
    Height = cam_params.size[1]

    f.write('%s %d %d %d %d %d %d\n' % (photo_params.name, 0, 0, 0, 0, Width, Height))

    f.close()


def write_params_for_colmap(cams_txt, image_txt, points_txt, cameras_params, photo_params, points=None):
    dir = os.path.dirname(cams_txt)
    if not os.path.exists(dir):
        os.mkdir(dir)

    # Camera list with one line of data per camera:
    # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Number of cameras: 3
    # 1 SIMPLE_PINHOLE f x cy
    # 2 PINHOLE 3072 2304 fx fy cx cy 2560.56 2560.56 1536 1152
    # 3 SIMPLE_RADIAL f cx cy k
    # 4 RADIAL f cx cy k1 k2
    # 5 OPENCV fx fy cx cy k1 k2 p1 p2
    # 6 FULL_OPENCV fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6

    # write cameras.txt
    cam_num = len(cameras_params)
    print("total %d cameras." % cam_num)
    f = open(cams_txt, "w")
    f.write('# Camera list with one line of data per camera:\n')
    f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy]\n')
    f.write('# Number of cameras: %d\n' % cam_num)
    for cam in cameras_params:
        f.write('%d %s %d %d %.6f %.6f %.6f %.6f' % (
        cam.camera_id, cam.camera_model, cam.size[0], cam.size[1], cam.focallength[0], cam.focallength[1], cam.x0y0[0],
        cam.x0y0[1]))
        if cam.camera_model == 'PINHOLE':
            f.write('\n')
        elif cam.camera_model == 'OPENCV':
            f.write(' %.6f %.6f %.6f %.6f' % (cam.distortion[0], cam.distortion[1], cam.distortion[2], cam.distortion[3]))
            f.write('\n')
        else:
            print('{}? Not implemented yet!'.format(cam.camera_model))

    f.close()

    photo_num = len(photo_params)
    print("total %d images." % photo_num)
    f = open(image_txt, "w")
    f.write("# Image list with two lines of data per image:\n")
    f.write('#  IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
    f.write('# Number of images: %d, mean observations per image: %f\n' % (photo_num, photo_num))

    for photo in photo_params:
        f.write('%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %s %s\n' % (
        photo[0], photo[1], photo[2], photo[3], photo[4], photo[5], photo[6], photo[7], photo[8], photo[9]))
        f.write('\n')
    f.close()

    f = open(points_txt, "w")
    f.write("# 3D point list with one line of data per point:\n")
    f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
    f.write('# Number of points: 0, mean track length: 0.0\n')
    f.close()


def write_params_for_predef(cams_txt, image_txt, cameras_params, photo_params):

    dir = os.path.dirname(cams_txt)
    if not os.path.exists(dir):
        os.mkdir(dir)

    # write cameras_info.txt
    if cameras_params:
        cam_num = len(cameras_params)
        f = open(cams_txt, "w")
        f.write('# Number of cameras: %d\n' % cam_num)
        f.write('# CAMERA_MODEL: OPENCV\n')
        f.write('# Camera list with one line of data per camera:\n')
        f.write('# CAMERA_ID, WIDTH, HEIGHT, PIXELSIZE, PARAMS[fx,fy,cx,cy], DISTORTION[K1, K2, P1, P2]\n')

        for cam in cameras_params:
            f.write('%d %d %d ' % (cam.camera_id, cam.size[0], cam.size[1]))
            f.write('%.6f %.6f %.6f %.6f %.6f ' % (cam.pixelsize, cam.focallength[0], cam.focallength[1], cam.x0y0[0],
                                                   cam.x0y0[1])) 
            f.write('%.6f %.6f %.6f %.6f\n' % (
            cam.distortion[0], cam.distortion[1], cam.distortion[2], cam.distortion[3])) 
        f.close()

    # write images_info.txt
    if photo_params:
        photo_num = len(photo_params)
        f = open(image_txt, "w")
        f.write('# Number of images: %d\n' % (photo_num))
        f.write("# Image list with two lines of data per image:\n")
        f.write("# CAMERA ORI: [ XrightYup | Rwc | twc ]\n")
        f.write('#  IMAGE_ID, CAMERA_ID, Rwc[9], twc[3], MINDEPTH, MAXDEPTH, NAME\n')

        for photo in photo_params:
            f.write('%d %d ' % (photo.image_id, photo.camera_id))
            f.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f ' % (
            photo.rotation_matrix[0, 0], photo.rotation_matrix[0, 1], photo.rotation_matrix[0, 2],
            photo.rotation_matrix[1, 0], photo.rotation_matrix[1, 1], photo.rotation_matrix[1, 2],
            photo.rotation_matrix[2, 0], photo.rotation_matrix[2, 1], photo.rotation_matrix[2, 2]))
            f.write('%.6f %.6f %.6f ' % (photo.project_center[0], photo.project_center[1], photo.project_center[2]))
            f.write('%.6f %.6f ' % (photo.depth[0], photo.depth[1]))
            f.write('%s\n' % (photo.name))
        f.close()


def read_images_path_text(path):
    paths_list = {}
    names_list = {}
    cluster_list = open(path).read().split()
    total_num = int(cluster_list[0])

    for i in range(total_num):
        index = int(cluster_list[i * 3 + 1]) 
        name = cluster_list[i * 3 + 2]
        p = cluster_list[i * 3 + 3]

        paths_list[index] = p
        names_list[index] = name

    return paths_list, names_list


def write_images_and_cameras_path_text(image_paths, cam_paths, image_path_list, cam_path_list):

    assert len(image_path_list) == len(cam_path_list)
    print(len(image_path_list))
    cnt1 = 1
    if image_path_list:
        f = open(image_paths, "w")
        f.write('%d\n' % len(image_path_list))
        for image_name, image_path in image_path_list:
            f.write('%d ' % cnt1)
            f.write('%s ' % image_name)
            f.write('%s\n' % image_path)
            cnt1 = cnt1 + 1
        f.close()

    cnt2 = 1
    if cam_path_list:
        f = open(cam_paths, "w")
        f.write('%d\n' % len(cam_path_list))
        for image_name, cam_path in cam_path_list:
            f.write('%d ' % cnt2)
            f.write('%s ' % image_name)
            f.write('%s\n' % cam_path)
            cnt2 = cnt2 + 1
        f.close()


def read_name_list(path):
    paths_list = {}
    names_list = {}
    cluster_list = open(path).read().split()
    total_num = int(cluster_list[0])

    for i in range(total_num):
        index = int(cluster_list[i * 3 + 1])
        name = cluster_list[i * 3 + 2]
        p = cluster_list[i * 3 + 3]

        paths_list[index] = p
        names_list[index] = name

    return names_list

# view pair
def read_view_pair_text(pair_path, view_num):

    metas = []
    # read the pair file
    with open(pair_path) as f:
        num_viewpoint = int(f.readline())
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                if len(src_views) < view_num:
                    print("{}< num_views:{}".format(len(src_views), view_num))
                    src_views += [src_views[0]] * (view_num - len(src_views))
                metas.append((ref_view, src_views))

    return metas


def read_view_pair_text_with_name(pair_path, pair_name_list, view_num):
    metas = []
    # read the pair file
    with open(pair_path) as f:
        num_viewpoint = int(f.readline())
        print("====>total {} ref_views.".format(num_viewpoint))
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            ref_view_name = os.path.splitext(pair_name_list[ref_view])[0]
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            src_views_name = [os.path.splitext(pair_name_list[src_view])[0] for src_view in src_views]

            if len(src_views_name) > 0:
                if len(src_views_name) < view_num:
                    print("{}< num_views:{}".format(len(src_views_name), view_num))
                    src_views_name += [src_views_name[0]] * (view_num - len(src_views_name))
                metas.append((ref_view_name, src_views_name)) 

    return metas


def write_pair_text(txt_file, score):
    text = "{}\n".format(len(score))
    for pair in score:
        text += "{}\n{} ".format(pair[0], len(pair[1]))
        for s in pair[1]:
            text += "{} {:.4f} ".format(s[0], s[1])
        text += "\n"

    with open(txt_file, "w") as f:
        f.write(text)


# scene block border
def write_block_text(txt_file, block_list):
    text = "{}\n".format(len(block_list))

    for block in block_list:
        sub_range, reference_images_list = block[0], block[1]
        for r in sub_range:
            text += "{:.4f} ".format(r)
        text += "\n"

        for i in reference_images_list:
            text += "{} ".format(i)
        text += "\n"

    with open(txt_file, "w") as f:
        f.write(text)


def save_border_as_file(fname, border):
    text = ""
    for b in border:
        text += str(b) + "\n"
    with open(fname, "w") as f:
        f.write(text)


def load_border_from_file(border_path):
    with open(border_path, "r") as f:
        text = f.read().splitlines()

    return np.array(text[0:6], dtype=np.float)


def write_center_offset(offset_path, offset):

    dir = os.path.dirname(offset_path)
    if not os.path.exists(dir):
        os.mkdir(dir)

    if offset:
        f = open(offset_path, "w")
        f.write('# Center offset\n')
        for cam in offset:
            f.write("{}\n".format(cam))
        f.close()


