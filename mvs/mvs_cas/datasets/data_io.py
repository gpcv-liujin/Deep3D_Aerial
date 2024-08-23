
from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import numpy as np
import re
import sys
import math
import cv2
import gdal
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import matplotlib.pyplot as plt


class Camera:
    def __init__(self, camera_id=None, size=None, pixelsize=None, focallength=None, x0y0=None, distortion=None):
        self.camera_id = camera_id  # id
        self.size = size  # [width, height]
        self.pixelsize = pixelsize  # px
        self.focallength = focallength  # [fx, fy]
        self.x0y0 = x0y0  # [x0, y0]
        self.distortion = distortion  # [k1, k2, k3, p1, p2]

    def __lt__(self):
        return [self.camera_id, self.size, self.pixelsize, self.focallength, self.x0y0, self.distortion]


class Photo:

    def __init__(self, image_id=None, camera_id=None, rotation_matrix=None, project_center=None, depth=None, name=None, camera_coordinate_type='XrightYup', rotation_type='Rwc', translation_type='twc'):
        self.image_id = image_id  
        self.camera_id = camera_id  
        self.name = name  
        self.rotation_matrix = rotation_matrix 
        self.project_center = project_center 
        self.depth = depth 
        self.camera_coordinate_type = camera_coordinate_type
        self.rotation_type = rotation_type
        self.translation_type = translation_type

    def __lt__(self):
        return [self.image_id, self.camera_id, self.rotation_matrix, self.project_center, self.depth, self.name]


def read_cameras_text(path):
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
                cams[camera_id] = Camera(camera_id=camera_id, size=[width, height], pixelsize=pixelsize,
                                         focallength=[params[0], params[1]], x0y0=[params[2], params[3]],
                                         distortion=distortion)

    return cams


def read_images_text(path):
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

                images[image_id] = Photo(image_id=image_id, camera_id=camera_id, rotation_matrix=R_matrix,
                                         project_center=t_matrix, depth=depth_range, name=image_name)

    return images


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


def read_view_pair_text(pair_path, view_num):

    metas = []
    with open(pair_path) as f:
        num_viewpoint = int(f.readline())
        # viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = [int(f.readline().rstrip())]
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                if len(src_views) < view_num:
                    print("{}< num_views:{}".format(len(src_views), view_num))
                    src_views += [src_views[0]] * (view_num - len(src_views))
                metas.append(ref_view + src_views)

    return metas


def Read_Img_Tone(path, x_lu, y_lu, x_size, y_size):
    dataset = gdal.Open(path)
    if dataset == None:
        print("GDAL RasterIO Error: Opening" + path + " failed!")
        return

    img = dataset.ReadAsArray(x_lu, y_lu, x_size, y_size)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)

    del dataset

    return img


def GetSize(path):
    dataset = gdal.Open(path)
    if dataset == None:
        print("GDAL RasterIO Error: Opening" + path + " failed!")
        return

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    del dataset
    return width, height


# pfm
def load_pfm_utf8(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm_utf8(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def read_pfm_latin1(fname):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    file = open(fname, 'rb')
    header = str(file.readline().decode('latin-1')).rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float((file.readline().decode('latin-1')).rstrip())
    if scale < 0:  # little-endian
        data_type = '<f'
    else:
        data_type = '>f'  # big-endian

    data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data


def write_pfm_latin1(file, image, scale=1):
    file = open(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale))

    image_string = image.tostring()
    file.write(image_string)

    file.close()


def write_red_cam(file, cam, location, ref_path):
    f = open(file, "w")

    f.write('extrinsic: XrightYdown, [Rcw|tcw]\n')
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
    f.write('\n')

    for word in location:
        f.write(str(word) + ' ')
    f.write(str(ref_path) + '\n')

    f.close()


def write_predef_cam(file, cam, location):
    pass



