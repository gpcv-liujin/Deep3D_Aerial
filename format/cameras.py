# Copyright © 2021 Jin Liu & Jian Gao.
# All rights reserved.
# The Group of Photogrammetry and Computer Vision
# The school of remote sensing and information engineering
# Wuhan University, Hubei, China

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
# from pyquaternion import Quaternion

"""
ORotation is a 3 × 3 rotation matrix bringing the camera axes defined by camera orientation 
to the canonical camera axes (x-axis oriented to the right side of the image, y-axis oriented
 to the bottom of the image, and z-axis oriented to the front of the camera)；
Reference: https://docs.bentley.com/LiveContent/web/ContextCapture%20Help-v10/en/GUID-2D452A8A-A4FE-450D-A0CA-9336DCF1238A.html
"""

ORotation = dict()
ORotation["xrightydown"] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
ORotation["xleftydown"] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float)
ORotation["xleftyup"] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float)
ORotation["xrightyup"] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
ORotation["xdownyright"] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float)
ORotation["xdownyleft"] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float)
ORotation["xupyleft"] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float)
ORotation["xupyright"] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float)


class Camera:

    def __init__(self, camera_id=None, camera_model=None, size=None, pixelsize=None, focallength=None, x0y0=None, distortion=None):
        self.camera_id = camera_id  # id int
        self.camera_model = camera_model
        self.size = size  # [width, height] int
        self.pixelsize = pixelsize  # px float
        self.focallength = focallength  # [fx, fy]  float
        self.x0y0 = x0y0  # [x0, y0]
        self.distortion = distortion  # [k1, k2, k3, p1, p2]

    def __lt__(self):
        return [self.camera_id, self.camera_model, self.size, self.pixelsize, self.focallength, self.x0y0, self.distortion]


class Photo:

    def __init__(self, image_id=None, camera_id=None, rotation_matrix=None, project_center=None, depth=None, name=None, camera_coordinate_type='XrightYup', rotation_type='Rwc', translation_type='twc'):
        self.image_id = image_id  # id int
        self.camera_id = camera_id  # id int
        self.name = name  # name str
        self.rotation_matrix = rotation_matrix  # Rwc [3,3] float
        self.project_center = project_center  # twc [x,y,z] float
        self.depth = depth  # [mindepth, maxdepth] float
        self.camera_coordinate_type = camera_coordinate_type
        self.rotation_type = rotation_type
        self.translation_type = translation_type

    def __lt__(self):
        return [self.image_id, self.camera_id, self.rotation_matrix, self.project_center, self.depth, self.name]


class toCamera:
    def __init__(self, rotmax, tvec, camera_coordinate_type="XRightYDown",
                 rotation_type="Rcw", translation_tye="tcw"):
        """
        :param rotmax:      rotation matrix
        :param tvec:        translation vector
        :param camera_coordinate_type: one of the 8 types in following:
                                    (1) XRightYDown (2) XLeftYDown (3) XLeftYUp
                                    (4) XRightYUp (5) XDownYRight (6) XDownYLeft
                                    (7) XUpYLeft  (8) xupyright
        :param rotation_type: Rcw or Rwc
        :param translation_tye: tcw or twc
        """
        self.rotation_matrix = rotmax
        self.translation_vector = np.reshape(tvec, (-1, 1))

        self.camera_coordinate_type = camera_coordinate_type
        assert self.search_in_list(
            camera_coordinate_type, list(ORotation.keys())) != 1, \
            "camera type must be one of the 7 types in " \
            "following: (1) XRightYDown (2) XLeftYDown" \
            " (3) XLeftYUp (4) XRightYUp (5) XDownYRight" \
            " (6) XDownYLeft (7) XUpYLeft"

        self.OMatrix = ORotation[camera_coordinate_type.lower()]
        self.rotation_type = rotation_type
        self.translation_type = translation_tye

        assert self.rotation_type == "Rwc" or self.rotation_type == "Rcw", \
            "rotation type must be Rwc or Rcw "
        assert self.translation_type == "twc" or self.translation_type == "tcw", \
            "rotation type must be twc or tcw "

        if self.rotation_type == "Rwc":
            self.rwc = self.rotation_matrix
            self.rcw = np.linalg.inv(self.rotation_matrix)
        else:
            self.rcw = self.rotation_matrix
            self.rwc = np.linalg.inv(self.rotation_matrix)

        if self.translation_type == "twc":
            self.tcw = - np.matmul(self.rcw, self.translation_vector)
            self.twc = self.translation_vector
        else:
            self.tcw = self.translation_vector
            self.twc = - np.matmul(self.rwc, self.translation_vector)

    @staticmethod
    def search_in_list(element_to_find, search_string_list):
        for idx, element in zip(range(len(search_string_list)), search_string_list):
            if element.upper() == element_to_find.upper():
                return idx
        return -1

    @classmethod
    def to_camera_cw_xright_ydown(cls, camera):
        """
        transform to the camera with Rcw, Tcw and XrightYDown coordinate.
        :return:
        """
        rotation_matrix = np.matmul(camera.OMatrix, camera.rcw)
        translation_vector = np.matmul(camera.OMatrix, camera.tcw)

        return cls(rotation_matrix, translation_vector)

    @classmethod
    def to_camera_wc_xright_yup(cls, camera):
        """
        transform to the camera with Rwc, Twc and XrightYUp coordinate.
        :return:
        """
        O_xrightyup = ORotation["xrightyup"]
        rotation_matrix = np.matmul(camera.rwc, O_xrightyup)
        translation_vector = camera.twc

        return cls(rotation_matrix, translation_vector)


def rot2quat(rot):
    r = R.from_dcm(rot)
    qvec_nvm = r.as_quat()
    quat_ = np.array(qvec_nvm)
    quat = [quat_[0], quat_[1], quat_[2], quat_[3]]

    return quat


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class testCamera:
    def __init__(self, k, rotmax, tvec, width, height,
                 camera_coordinate_type="XRightYDown",
                 rotation_type="Rcw", translation_tye="tcw"):
        """
        :param k:           intrinsic
        :param rotmax:      rotation matrix
        :param tvec:        translation vector
        :param width:       the width of the image
        :param height:      the height of the image
        :param camera_coordinate_type: one of the 8 types in following:
                                    (1) XRightYDown (2) XLeftYDown (3) XLeftYUp
                                    (4) XRightYUp (5) XDownYRight (6) XDownYLeft
                                    (7) XUpYLeft  (8) xupyright
        :param rotation_type: Rcw or Rwc
        :param translation_tye: tcw or twc
        """
        self.rotation_matrix = rotmax
        self.translation_vector = np.reshape(tvec, (-1, 1))
        self.intrinsic = k
        self.width = width
        self.height = height

        self.camera_coordinate_type = camera_coordinate_type
        assert self.search_in_list(
            camera_coordinate_type, list(ORotation.keys())) != 1, \
            "camera type must be one of the 7 types in " \
            "following: (1) XRightYDown (2) XLeftYDown" \
            " (3) XLeftYUp (4) XRightYUp (5) XDownYRight" \
            " (6) XDownYLeft (7) XUpYLeft"

        self.OMatrix = ORotation[camera_coordinate_type.lower()]
        self.rotation_type = rotation_type
        self.translation_type = translation_tye

        assert self.rotation_type == "Rwc" or self.rotation_type == "Rcw", \
            "rotation type must be Rwc or Rcw "
        assert self.translation_type == "twc" or self.translation_type == "tcw", \
            "rotation type must be twc or tcw "
        if self.rotation_type == "Rwc":
            self.rwc = self.rotation_matrix
            self.rcw = np.linalg.inv(self.rotation_matrix)
        else:
            self.rcw = self.rotation_matrix
            self.rwc = np.linalg.inv(self.rotation_matrix)

        if self.translation_type == "twc":
            self.tcw = - np.matmul(self.rcw, self.translation_vector)
            self.twc = self.translation_vector
        else:
            self.tcw = self.translation_vector
            self.twc = - np.matmul(self.rwc, self.translation_vector)

    @staticmethod
    def search_in_list(element_to_find, search_string_list):
        for idx, element in zip(range(len(search_string_list)), search_string_list):
            if element.upper() == element_to_find.upper():
                return idx
        return -1

    @classmethod
    def read_from_text(cls, text_file):
        assert os.path.exists(text_file), ("{} does not exist".format(text_file))

        with open(text_file, "r") as f:
            data = f.read().splitlines()

        coordinate_info_idx = cls.search_in_list("COORDINATE INFO", data)
        if coordinate_info_idx != -1:
            camera_coordinate_type, rotation_type, translation_type = data[coordinate_info_idx + 1].split()
        else:
            camera_coordinate_type = "XRightYDown"
            rotation_type = "Rcw"
            translation_type = "tcw"

        intrinsic_info_idx = cls.search_in_list("INTRINSIC", data)
        assert intrinsic_info_idx != -1, ("No intrinsic information is found in this file: " + text_file)
        kmat = np.array([d.split() for d in data[intrinsic_info_idx + 1:intrinsic_info_idx + 4]], dtype=np.float)

        extrinsic_info_idx = cls.search_in_list("EXTRINSIC", data)
        assert extrinsic_info_idx != -1, ("No extrinsic information is found in this file: " + text_file)
        emat = np.array([d.split() for d in data[extrinsic_info_idx + 1:extrinsic_info_idx + 5]], dtype=np.float)
        rmat = emat[0:3, 0:3]
        tvec = emat[0:3, 3]

        size_info_idx = cls.search_in_list("WIDTH HEIGHT", data)
        assert size_info_idx != -1, ("No size information is found in this file: " + text_file)
        size = np.array(data[size_info_idx + 1].split(), dtype=np.int)

        return cls(kmat, rmat, tvec, size[0], size[1], camera_coordinate_type, rotation_type, translation_type)

    @classmethod
    def to_camera_cw_xright_ydown(cls, camera):
        """
        transform to the camera with Rcw, Tcw and XrightYDown coordinate.
        :return:
        """
        rotation_matrix = np.matmul(camera.OMatrix, camera.rcw)
        translation_vector = np.matmul(camera.OMatrix, camera.tcw)

        return cls(camera.intrinsic, rotation_matrix, translation_vector, camera.width, camera.height)

    @classmethod
    def to_camera_wc_xright_yup(cls, camera):
        """
        transform to the camera with Rwc, Twc and XrightYUp coordinate.
        :return:
        """
        O_xrightyup = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        rotation_matrix = np.matmul(camera.rwc, O_xrightyup)
        translation_vector = camera.twc

        return cls(camera.intrinsic, rotation_matrix, translation_vector, camera.width, camera.height)

    def world2image(self, X, Y, Z):
        """
        :param pts: [n, 3]
        :return: image pts [n, 2], depth [n, 1]
        """
        Xarray = np.array(X)
        Yarray = np.array(Y)
        Zarray = np.array(Z)

        assert Xarray.shape == Yarray.shape and Yarray.shape == Zarray.shape, \
            ("Invalid input, the sizes are invalid"
             " X({}), Y({}), Z({})".format(Xarray.shape, Yarray.shape, Zarray.shape))

        shape = Xarray.shape
        pts = np.stack([Xarray.reshape(-1), Yarray.reshape(-1), Zarray.reshape(-1)], axis=-1)

        KOMatrix = np.matmul(self.intrinsic, self.OMatrix)
        KORcwMatrix = np.matmul(KOMatrix, self.rcw)
        u = np.matmul(KORcwMatrix, pts.T - self.twc).T
        u[:, 0:2] = u[:, 0:2] / u[:, 2:3]

        return u[:, 0].reshape(shape), u[:, 1].reshape(shape), u[:, 2].reshape(shape)

    def image2world(self, x, y, depth):
        """
        :param pts: [n, 2]
        :param depth: [n, 1]
        :return:
        """
        xarray = np.array(x)
        yarray = np.array(y)
        darray = np.array(depth)

        assert xarray.shape == yarray.shape and yarray.shape == darray.shape, \
            ("Invalid input, the sizes are invalid "
             "x({}), y({}), depth({})".format(xarray.shape, yarray.shape, darray.shape))

        shape = xarray.shape
        pts_with_depth = np.stack([xarray.reshape(-1) * darray.reshape(-1),
                                   yarray.reshape(-1) * darray.reshape(-1),
                                   darray.reshape(-1)], axis=-1)

        OinvKinvMatrix = np.linalg.inv(np.matmul(self.intrinsic, self.OMatrix))
        RwcOinvKinvMatrix = np.matmul(self.rwc, OinvKinvMatrix)
        Xw = (np.matmul(RwcOinvKinvMatrix, pts_with_depth.T) + self.twc).T

        return Xw[:, 0].reshape(shape), Xw[:, 1].reshape(shape), Xw[:, 2].reshape(shape)

    def write_as_text(self, text_file, precision=12):
        format_str = "{{:.{}f}}".format(precision)

        text_str = "COORDINATE INFO\n"
        text_str += "{} {} {}\n".format(self.camera_coordinate_type, self.rotation_type, self.translation_type)

        text_str += "\nINTRINSIC\n"
        for i in range(3):
            for j in range(3):
                text_str += format_str.format(self.intrinsic[i][j]) + " "
            text_str += "\n"

        extrinsic = np.eye(4, dtype=np.float)
        extrinsic[0:3, 0:3] = self.rotation_matrix
        extrinsic[3, 0:3] = self.translation_vector.reshape(-1)

        text_str += "\nEXTRINSIC\n"
        for j in range(4):
            for i in range(4):
                text_str += format_str.format(extrinsic[i][j]) + " "
            text_str += "\n"

        text_str += "\nWIDTH HEIGHT\n"
        text_str += "{} {}\n".format(self.width, self.height)

        with open(text_file, "w") as f:
            f.write(text_str)


def test():
    import cv2
    import matplotlib.pyplot as plt

    path1 = "test/cams/001_001.txt"
    path2 = "test/cams/001_002.txt"

    camera1 = testCamera.read_from_text(path1)
    camera2 = testCamera.read_from_text(path2)

    img1 = cv2.imread("test/imgs/001_1.png", cv2.IMREAD_UNCHANGED)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.imread("test/imgs/001_2.png", cv2.IMREAD_UNCHANGED)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    x1 = [3056]
    y1 = [2440]
    depth_map = cv2.imread("test/depths/001_1.png", cv2.IMREAD_ANYDEPTH)
    depth1 = [depth_map[y1[0], x1[0]] / 64]

    X, Y, Z = camera1.image2world(x1, y1, depth1)
    x2, y2, depth2 = camera2.world2image(X, Y, Z)

    print(x1, y1)
    print(x2, y2)

    for i in range(1, 501, 20):
        cv2.circle(img1, (int(x1[0]), int(y1[0])), i, (0, 255, 0), 2)
        cv2.circle(img2, (int(x2[0]), int(y2[0])), i, (0, 255, 0), 2)

    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.subplot(1, 3, 3)
    plt.imshow(depth_map)
    plt.show()


if __name__ == "__main__":
    test()
