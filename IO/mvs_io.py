#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2022, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu


import os
import struct
import numpy as np
import sys
import collections
import pandas as pd
from pyntcloud import PyntCloud

def saveString(s, fid):
    length = len(s)
    if length > 0:
        string = struct.pack(str(length) + 's', s.encode("utf-8"))
        fid.write(string)
def saveUint(uint, fid):
    data = struct.pack('I', uint)
    fid.write(data)
def saveFloat(f, fid):
    data = struct.pack("f", f)
    fid.write(data)
def saveDouble(d, fid):
    data = struct.pack("d", d)
    fid.write(data)
def saveUint64(l, fid):
    data = struct.pack("q", l)
    fid.write(data)
def saveUShort(us, fid):
    data = struct.pack("H", us)
    fid.write(data)
def saveUchar(uc, fid):
    data = struct.pack("B", uc)
    fid.write(data)
def saveMatrix(A, fid, fmt):
    h, w = A.shape[0], A.shape[1]
    size = h * w
    data = struct.pack(fmt * size, *(A.reshape(-1)))
    fid.write(data)

def saveMatrix3D(A, fid, fmt):
    h, w, c = A.shape[0], A.shape[1], A.shape[2]
    size = h * w * c
    data = struct.pack(fmt * size, *(A.reshape(-1)))
    fid.write(data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)

def loadString(fid, length):
    if length <= 0:
        return ""
    return fid.read(length).decode("utf-8")
def loadUint(fid):
    data = fid.read(4)
    data = struct.unpack("I", data)[0]
    return data
def loadFloat(fid):
    data = fid.read(4)
    data = struct.unpack("f", data)[0]
    return data
def loadDouble(fid):
    data = fid.read(8)
    data = struct.unpack("d", data)[0]
    return data
def loadUint64(fid):
    data = fid.read(8)
    data = struct.unpack("Q", data)[0]
    return data
def loadUShort(fid):
    data = fid.read(2)
    data = struct.unpack("H", data)[0]
    return data
def loadUchar(fid):
    data = fid.read(1)
    data = struct.unpack("B", data)[0]
    return data
def loadMatrix(fid, rows, cols, fmt):
    size = rows * cols
    if fmt == 'f':
        data = fid.read(4 * size)
        data = np.array(struct.unpack("f" * size, data))
    elif fmt == 'd':
        data = fid.read(8 * size)
        data = np.array(struct.unpack("d" * size, data))
    elif fmt == 'i' or fmt == 'I' or fmt == 'l' or fmt == 'L':
        data = fid.read(4 * size)
        data = np.array(struct.unpack(fmt * size, data))
    elif fmt == 'q' or fmt == 'Q':
        data = fid.read(8 * size)
        data = np.array(struct.unpack(fmt * size, data))
    elif fmt == 'b' or fmt == 'B':
        data = fid.read(1 * size)
        data = np.array(struct.unpack(fmt * size, data))
    A = data.reshape(rows, cols)

    return A
def loadMatrix3D(fid, rows, cols, channels, fmt):
    size = rows * cols * channels
    if fmt == 'f':
        data = fid.read(4 * size)
        data = np.array(struct.unpack("f" * size, data))
    elif fmt == 'd':
        data = fid.read(8 * size)
        data = np.array(struct.unpack("d" * size, data))
    elif fmt == 'i' or fmt == 'I' or fmt == 'l' or fmt == 'L':
        data = fid.read(4 * size)
        data = np.array(struct.unpack(fmt * size, data))
    elif fmt == 'q' or fmt == 'Q':
        data = fid.read(8 * size)
        data = np.array(struct.unpack(fmt * size, data))
    elif fmt == 'b' or fmt == 'B':
        data = fid.read(1 * size)
        data = np.array(struct.unpack(fmt * size, data))
    A = data.reshape(rows, cols, channels)

    return A

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


class Camera(object):
    def __init__(self):
        self.width = 0
        self.height = 0
        self.name = ''
        self.bandName = ''
        self.K = np.eye(3)
        self.R = np.eye(3)
        self.C = np.array([0., 0., 0.], dtype='float')

    def HasResolution(self):
        return (self.width > 0 and self.height > 0)
    def IsNormalized(self):
        return not self.HasResolution()
    def GetNormalizationScale(self):
        return np.max([self.height, self.width])
    def NormalizeIntrinsics(self):
        scale = 1.0 / self.GetNormalizationScale()
        self.K[0, 0] *= scale
        self.K[1, 1] *= scale
        self.K[0, 2] = (self.K[0, 2] + 0.5) * scale - 0.5
        self.K[1, 2] = (self.K[1, 2] + 0.5) * scale - 0.5

    def Save(self, fid, version = 6):
        saveUint64(len(self.name), fid)
        saveString(self.name, fid)
        if version > 3:
            # bandName
            saveUint64(len(self.bandName), fid)
            saveString(self.bandName, fid)
        if version > 0:
            saveUint(self.width, fid)
            saveUint(self.height, fid)
        saveMatrix(self.K, fid, "d")
        saveMatrix(self.R, fid, "d")
        saveMatrix(self.C.reshape(3, 1), fid, "d")

    def Load(self, fid, version = 6):
        name_size = loadUint64(fid)
        if name_size > 0:
            self.name = loadString(fid, name_size)

        if version > 3:
            bandname_size = loadUint64(fid)
            self.bandName = loadString(fid, bandname_size)

        if version > 0:
            self.width = loadUint(fid)
            self.height = loadUint(fid)

        self.K = loadMatrix(fid, 3, 3, 'd')
        self.R = loadMatrix(fid, 3, 3, 'd')
        self.C = loadMatrix(fid, 3, 1, 'd')

class Pose(object):
    def __init__(self):
        self.R = np.eye(3)
        self.C = np.array([0., 0., 0.]).reshape(3, 1)
    def Save(self, fid):
        saveMatrix(self.R, fid, "d")
        saveMatrix(self.C.reshape(3, 1), fid, "d")
    def Load(self, fid):
        self.R = loadMatrix(fid, 3, 3, "d")
        self.C = loadMatrix(fid, 3, 1, "d")

class Platform(object):
    def __init__(self):
        self.name = ""
        self.cameras = []
        self.poses = []
    def GetPose(self, cameraID, poseID):
        camera = self.cameras[cameraID]
        pose = self.poses[poseID]
        R = camera.R @ pose.R
        t = pose.R.transpose() @ camera.C + pose.C
        p = Pose()
        p.R = R
        p.C = t
        return p
    def GetFullK(self, cameraID, width, height):
        camera = self.cameras[cameraID]
        if (not camera.IsNormalized()) and camera.width == width and camera.height == height:
            return camera.K
        else:
            scale = 1
            if camera.IsNormalized():
                scale = camera.GetNormalizationScale()
            K = camera.K
            K[0, 0] *= scale
            K[1, 1] *= scale
            K[0, 2] = (K[0, 2] + 0.5) * scale - 0.5
            K[1, 2] = (K[1, 2] + 0.5) * scale - 0.5
            K[0, 1] *= scale
            return K

    def Save(self, fid, version = 6):
        saveUint64(len(self.name), fid)
        saveString(self.name, fid)

        saveUint64(len(self.cameras), fid)
        for camera in self.cameras:
            camera.Save(fid, version)

        saveUint64(len(self.poses), fid)
        for pose in self.poses:
            pose.Save(fid)

    def Load(self, fid, version = 6):
        name_size = loadUint64(fid)
        if name_size > 0:
            self.name = loadString(fid, name_size)

        camera_size = loadUint64(fid)
        if camera_size > 0:
            for i in range(camera_size):
                camera = Camera()
                camera.Load(fid, version)
                self.cameras.append(camera)

        pose_size = loadUint64(fid)
        if pose_size > 0:
            for i in range(pose_size):
                pose = Pose()
                pose.Load(fid)
                self.poses.append(pose)

class Image(object):
    def __init__(self):
        self.name = ""
        self.maskName = ""
        self.platformID = -1
        self.cameraID = -1
        self.poseID = -1
        self.ID = -1
    def Save(self, fid, version = 6):
        saveUint64(len(self.name), fid)
        saveString(self.name, fid)
        if version > 4:
            saveUint64(len(self.maskName), fid)
            saveString(self.maskName, fid)
        saveUint(self.platformID, fid)
        saveUint(self.cameraID, fid)
        saveUint(self.poseID, fid)
        if version > 2:
            saveUint(self.ID, fid)
    def Load(self, fid, version):
        name_length = loadUint64(fid)
        self.name = loadString(fid, name_length)
        if version > 4:
            maskName_length = loadUint64(fid)
            self.maskName = loadString(fid, maskName_length)
        self.platformID = loadUint(fid)
        self.cameraID = loadUint(fid)
        self.poseID = loadUint(fid)
        if version > 2:
            self.ID = loadUint(fid)

class View(object):
    def __init__(self, imageID, confidence):
        self.imageID = imageID
        self.confidence = confidence

class Vertex(object):
    def __init__(self):
        self.X = np.zeros((3, 1), dtype="float")
        self.views = []
        self.confidence = []
    def viewSort(self):
        self.views.sort()

    def Save(self, fid):
        saveMatrix(self.X.reshape((3, 1)), fid, "f")
        saveUint64(len(self.confidence), fid)
        for i in range(len(self.confidence)):
            saveUint(self.views[i], fid)
            saveFloat(self.confidence[i], fid)

    def Load(self, fid):
        self.X = loadMatrix(fid, 3, 1, "f")
        view_size = loadUint64(fid)
        for i in range(view_size):
            view = loadUint(fid)
            confidence = loadFloat(fid)
            self.views.append(view)
            self.confidence.append(confidence)

class Line(object):
    def __init__(self):
        self.X1 = np.array([0., 0., 0.])
        self.X2 = np.array([0., 0., 0.])
        self.views = []
        self.confidence = []

class OBB(object):
    def __init__(self):
        self.rot = np.eye(3, dtype='float64')
        self.ptMin = np.zeros((3, 1), dtype='float64')
        self.ptMax = np.zeros((3, 1), dtype='float64')
    def Save(self, fid):
        saveMatrix(self.rot, fid, "d")
        saveMatrix(self.ptMin, fid, "d")
        saveMatrix(self.ptMax, fid, "d")
    def Load(self, fid):
        self.rot = loadMatrix(fid, 3, 3, "d")
        self.ptMin = loadMatrix(fid, 3, 1, "d")
        self.ptMax = loadMatrix(fid, 3, 1, "d")


MeshPoint = collections.namedtuple(
    "MeshingPoint", ["position", "color", "normal", "num_visible_images", "visible_image_idxs"])
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])


class Interface():
    def __init__(self, sparse_path, image_path, save_path, input_data=True):

        self.sparse_path = sparse_path
        self.image_path = image_path
        print(self.image_path)
        self.fused_mvs_path = save_path

        (path, filename) = os.path.split(save_path)
        print(path, filename)
        if not os.path.exists(os.path.join(path, '1')):
            os.mkdir(os.path.join(path, '1'))
        self.fused_ply_path = os.path.join(path, '1/{}'.format(filename.replace('.mvs', '.ply')))

        self.sparse_camera_path = os.path.join(self.sparse_path, 'cameras.bin')
        self.sparse_image_path = os.path.join(self.sparse_path, 'images.bin')
        self.sparse_point_path = os.path.join(self.sparse_path, 'points3D.bin')

        self.bNormalizeIntrinsics = False

        self.platforms = []
        self.mapCamera = {}
        self.mapImages = {}
        self.images = []

        self.vertices = []
        self.verticesNormal = []
        self.verticesColor = []
        self.lines = []
        self.linesNormal = []
        self.linesColor = []

        self.transform = np.eye(4, dtype='float64')
        self.obb = OBB()

        self.readCamera()
        self.readImages()
        if not input_data:
            self.readSparsePoint()

    # TODO
    def Release(self):
        print()

    def readCamera(self):
        with open(self.sparse_camera_path, "rb") as fid:
            num_cameras = read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_cameras):
                camera_properties = read_next_bytes(
                    fid, num_bytes=24, format_char_sequence="iiQQ")
                camera_id = camera_properties[0] - 1
                model_id = camera_properties[1]
                model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
                width = camera_properties[2]
                height = camera_properties[3]
                num_params = CAMERA_MODEL_IDS[model_id].num_params
                params = read_next_bytes(fid, num_bytes=8 * num_params,
                                         format_char_sequence="d" * num_params)

                platform = Platform()
                platform.name = "platform" + str(camera_id).zfill(3)
                self.mapCamera[str(camera_id)] = len(self.platforms)  # 0-0
                camera = Camera()
                camera.name = model_name
                if camera.name.upper() != "PINHOLE":
                    print("The camera model is not PINHOLE")
                    # sys.exit()
                camera.K[0, 0] = params[0]
                camera.K[1, 1] = params[1]
                camera.K[0, 2] = params[2]
                camera.K[1, 2] = params[3]

                if self.bNormalizeIntrinsics:
                    camera.NormalizeIntrinsics()
                else:
                    camera.width = width
                    camera.height = height

                platform.cameras.append(camera)
                self.platforms.append(platform)

    def readImages(self):
        def qvec2rotmat(qvec):
            return np.array([
                [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
                [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
                [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])

        with open(self.sparse_image_path, "rb") as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_reg_images):
                binary_image_properties = read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi")
                # image_id = binary_image_properties[0]
                image_id = binary_image_properties[0] - 1
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                # camera_id = binary_image_properties[8]
                camera_id = binary_image_properties[8] - 1
                image_name = ""
                current_char = read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":  # look for the ASCII 0 entry
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(fid, 1, "c")[0]
                num_points2D = read_next_bytes(fid, num_bytes=8,
                                               format_char_sequence="Q")[0]
                x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                           format_char_sequence="ddq" * num_points2D)
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                       tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))

                pose = Pose()
                pose.R = qvec2rotmat(qvec)
                pose.C = -pose.R.transpose() @ tvec.transpose()

                image = Image()
                image.name = os.path.join(self.image_path, image_name)
                image.platformID = self.mapCamera[str(camera_id)]  # 0-0
                image.cameraID = 0
                image.ID = image_id  # 1-0
                image.poseID = len(self.platforms[image.platformID].poses)

                self.mapImages[str(image_id)] = len(self.images)
                self.platforms[image.platformID].poses.append(pose)
                self.images.append(image)

    def readSparsePoint(self):
        with open(self.sparse_point_path, "rb") as fid:
            num_points = read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_points):
                binary_point_line_properties = read_next_bytes(
                    fid, num_bytes=43, format_char_sequence="QdddBBBd")
                point3D_id = binary_point_line_properties[0] - 1
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = read_next_bytes(
                    fid, num_bytes=8, format_char_sequence="Q")[0]
                track_elems = read_next_bytes(
                    fid, num_bytes=8 * track_length,
                    format_char_sequence="ii" * track_length)
                image_ids = np.array(tuple(map(int, track_elems[0::2]))) - 1
                point2D_idxs = np.array(tuple(map(int, track_elems[1::2]))) - 1

                vertex = Vertex()
                vertex.X = xyz
                vertex.views = self.SparsePointIDTransfer(image_ids)
                vertex.viewSort()
                vertex.confidence = [0] * vertex.views.shape[0]

                self.vertices.append(vertex)
                self.verticesColor.append(rgb)

    def readDensePoint(self, fused_path, vis_path):
        if (not os.path.isfile(fused_path)) or (not os.path.isfile(vis_path)):
            print("No fused.ply or fused.ply.vis, begin to read sparse points")
            return False

        point_cloud = PyntCloud.from_file(fused_path)
        xyz_arr = point_cloud.points.loc[:, ["x", "y", "z"]].to_numpy()
        normal_arr = point_cloud.points.loc[:, ["nx", "ny", "nz"]].to_numpy()
        color_arr = point_cloud.points.loc[:, ["blue", "green", "red"]].to_numpy()

        with open(vis_path, "rb") as fid:
            num_points = read_next_bytes(fid, 8, "Q")[0]
            for i in range(num_points):
                num_visible_images = read_next_bytes(fid, 4, "I")[0]
                visible_image_idxs = read_next_bytes(
                    fid, num_bytes=4 * num_visible_images,
                    format_char_sequence="I" * num_visible_images)
                visible_image_idxs = np.array(tuple(map(int, visible_image_idxs)))

                vertex = Vertex()
                vertex.X = xyz_arr[i]
                vertex.views = visible_image_idxs
                vertex.viewSort()
                vertex.confidence = [0.0] * len(visible_image_idxs)

                self.vertices.append(vertex)
                self.verticesColor.append(color_arr[i])
                self.verticesNormal.append(normal_arr[i])

        return True

    def NameInImage(self, name):
        for i in range(len(self.images)):
            pictName = self.images[i].name.split('\\')[-1]
            if name == pictName:
                return i
        return -1

    def SparsePointIDTransfer(self, image_ids):
        image_ids_transfer = []
        keys = self.mapImages.keys()
        for image_id in image_ids:
            if str(image_id) in keys:
                image_ids_transfer.append(self.mapImages[str(image_id)])
        return np.array(image_ids_transfer)

    def writeMVS(self, version=6):
        with open(self.fused_mvs_path, "wb") as fid:
            # Header
            if version > 0:
                saveString("MVSI", fid)
                saveUint(version, fid)
                saveUint(0, fid)

            # platforms
            saveUint64(len(self.platforms), fid)
            for platform in self.platforms:
                platform.Save(fid, version)

            # Images
            saveUint64(len(self.images), fid)
            for image in self.images:
                image.Save(fid, version)

            # Vertices
            saveUint64(len(self.vertices), fid)
            for vertex in self.vertices:
                vertex.Save(fid)

            # Normal
            saveUint64(len(self.verticesNormal), fid)
            verticesNormal = np.asarray(self.verticesNormal)
            saveMatrix(verticesNormal, fid, 'f')

            # Color
            saveUint64(len(self.verticesColor), fid)
            verticesColor = np.asarray(self.verticesColor)
            saveMatrix(verticesColor, fid, 'B')

            # Save lines, linesNormal, linesColor
            saveUint64(0, fid)
            saveUint64(0, fid)
            saveUint64(0, fid)

            if version > 1:
                saveMatrix(self.transform, fid, "d")
            if version > 5:
                self.obb.Save(fid)

    def readMVS(self, MVSPath):
        with open(MVSPath, "rb") as fid:
            # read header
            szHeader = loadString(fid, 4)
            if szHeader != "MVSI":
                print("Error .mvs type")
                fid.close()
                return False

            version = loadUint(fid)
            if version > 6:
                print("Error .mvs type")
                fid.close()
                return False

            loadUint(fid)

            # Platforms
            platform_size = loadUint64(fid)
            for i in range(platform_size):
                platform = Platform()
                platform.Load(fid, version)
                self.platforms.append(platform)

            # Images
            image_size = loadUint64(fid)
            for i in range(image_size):
                image = Image()
                image.Load(fid, version)
                self.images.append(image)

            # vertices
            vertex_size = loadUint64(fid)
            if vertex_size > 0:
                for i in range(vertex_size):
                    vertex = Vertex()
                    vertex.Load(fid)
                    self.vertices.append(vertex)

            # verticesNormal
            verticesNormal_size = loadUint64(fid)
            if verticesNormal_size > 0:
                for i in range(verticesNormal_size):
                    vertexNormal = loadMatrix(fid, 3, 1, "f")
                    self.verticesNormal.append(vertexNormal)

            # verticesColor
            verticesColor_size = loadUint64(fid)
            if verticesColor_size > 0:
                for i in range(verticesColor_size):
                    vertexColor = loadMatrix(fid, 3, 1, "B")
                    self.verticesColor.append(vertexColor)

            if version > 0:
                loadUint64(fid)
                loadUint64(fid)
                loadUint64(fid)

            if version > 1:
                self.transform = loadMatrix(fid, 4, 4, "d")
            if version > 5:
                self.obb = OBB()
                self.obb.Load(fid)

    def writePLY(self):
        columns = ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"]
        points_data_frame = pd.DataFrame(np.zeros((len(self.vertices), len(columns))), columns=columns)

        positions = np.asarray([point.X for point in self.vertices])
        normals = np.asarray([normal for normal in self.verticesNormal])
        colors = np.asarray([color for color in self.verticesColor])

        points_data_frame.loc[:, ["x", "y", "z"]] = positions
        points_data_frame.loc[:, ["nx", "ny", "nz"]] = normals
        points_data_frame.loc[:, ["red", "green", "blue"]] = colors

        # print(positions.dtype, colors.dtype, normals.dtype)

        points_data_frame = points_data_frame.astype({
            "x": positions.dtype, "y": positions.dtype, "z": positions.dtype,
            "red": colors.dtype, "green": colors.dtype, "blue": colors.dtype,
            "nx": normals.dtype, "ny": normals.dtype, "nz": normals.dtype})

        # points_data_frame = points_data_frame.astype({
        #     "x": 'f4', "y": 'f4', "z": 'f4', "red": 'u1', "green": 'u1', "blue": 'u1'})
        point_cloud = PyntCloud(points_data_frame)
        point_cloud.to_file(self.fused_ply_path)

    def Interface_Fused(self, vertices, verticesColor, verticesNormal, save_ply=True, version=6):
        self.vertices.extend(vertices)
        self.verticesColor.extend(verticesColor)
        self.verticesNormal.extend(verticesNormal)
        self.writeMVS(version=version)
        if save_ply:
            self.writePLY()

    def Interface_Colmap(self, fused_path, vis_path, version=6):
        if not self.readDensePoint(fused_path, vis_path):
            self.readSparsePoint()
        self.writeMVS(version=version)




import argparse
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--image_dir',
                    default=r"G:\other_data\pipeline\workspace_tianjin_2_scale4_openmvs\mvs\dense\images",
                    help="the directory wich contains the pictures set.")
PARSER.add_argument('--project_dir',
                    default=r"G:\other_data\pipeline\workspace_tianjin_2_scale4_openmvs\mvs\dense",
                    help="the directory wich will contain the resulting files.")
PARSER.add_argument('--save_dir',
                    default="G:\other_data\pipeline\workspace_tianjin_2_scale4_openmvs\mvs\dense\scene_dense.mvs",
                    help="the directory wich will contain the mesh result for colmap.")
PARSER.add_argument('--params_dir',
                    default="G:\other_data\pipeline\workspace_tianjin_2_scale4_openmvs\mvs\sparse",
                    help="the directory wich will contain the mesh result for colmap.")
args = PARSER.parse_args()

if __name__ == '__main__':
    mvsp = Interface(args.params_dir, args.image_dir, args.save_dir)

    mvsp.readMVS(r'G:\other_data\pipeline\workspace_tianjin_2_scale4_openmvs\mvs\dense\scene_dense.mvs')
    mvsp.writePLY()











