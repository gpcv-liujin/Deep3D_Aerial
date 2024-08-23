#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2022, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu


import struct
import numpy as np


# save and load from bin file
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
    size = h*w
    data = struct.pack(fmt*size, *(A.reshape(-1)))
    fid.write(data)


def saveMatrix3D(A, fid, fmt):
    h, w, c = A.shape[0], A.shape[1], A.shape[2]
    size = h * w * c
    data = struct.pack(fmt*size, *(A.reshape(-1)))
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
        data = fid.read(4*size)
        data = np.array(struct.unpack("f"*size, data))
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



def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def ExportDepthDataRaw(savefileName, imageFileName, IDs, imgSize, K, R, C,
                       dMin, dMax, depthMap, normalMap, confMap):
    assert depthMap.size != 0
    assert confMap.size == 0 or confMap.shape == depthMap.shape
    assert depthMap.shape[0] <= imgSize[0] and depthMap.shape[1] <= imgSize[1]

    with open(savefileName, "wb") as f:
        # header
        headerName = 21060                       # ushort
        headerType = 1                           # uchar
        headerPadding = 0                        # uchar
        headerImgWidth = imgSize[1]              # uint
        headerImgHeight = imgSize[0]             # uint
        headerDepthWidth = depthMap.shape[1]     # uint
        headerDepthHeight = depthMap.shape[0]    # uint
        headerdMin = dMin                        # float
        headerdMax = dMax                        # float

        if normalMap.size != 0:
            headerType = headerType | (2)
        if confMap.size != 0:
            headerType = headerType | (4)

        saveUShort(headerName, f)
        saveUchar(headerType, f)
        saveUchar(headerPadding, f)
        saveUint(headerImgWidth, f)
        saveUint(headerImgHeight, f)
        saveUint(headerDepthWidth, f)
        saveUint(headerDepthHeight, f)
        saveFloat(headerdMin, f)
        saveFloat(headerdMax, f)

        # names
        saveUShort(len(imageFileName), f)
        saveString(imageFileName, f)

        # neighbors
        saveUint(len(IDs), f)
        for id in IDs:
            saveUint(id, f)

        # pose
        saveMatrix(K.reshape((3, 3)), f, "d")
        saveMatrix(R.reshape((3, 3)), f, "d")
        saveMatrix(C.reshape((3, 1)), f, "d")

        # depth-map
        saveMatrix(depthMap, f, 'f')

        # normal-map
        if (headerType & (2)) != 0:
            saveMatrix3D(normalMap, f, "f")

        # confidence-map
        if (headerType & (4) != 0):
            saveMatrix(confMap, f, 'f')



def ImportDepthDataRaw(fileName):
    with open(fileName, "rb") as f:
        # Header
        headerName = loadUShort(f)
        headerType = loadUchar(f)
        headerPadding = loadUchar(f)
        headerImgWidth = loadUint(f)
        headerImgHeight = loadUint(f)
        headerDepthWidth = loadUint(f)
        headerDepthHeight = loadUint(f)
        headerdMin = loadFloat(f)
        headerdMax = loadFloat(f)

        # names
        imgFileName_size = loadUShort(f)
        imageFileName = loadString(f, imgFileName_size)

        # neighbors
        IDs_size = loadUint(f)
        IDs = []
        for i in range(IDs_size):
            IDs.append(loadUint(f))

        # pose
        K = loadMatrix(f, 3, 3, "d")
        R = loadMatrix(f, 3, 3, "d")
        C = loadMatrix(f, 3, 1, "d")

        # depth-map
        depthMap = loadMatrix(f, headerDepthHeight, headerDepthWidth, "f")

        # normal-map
        if (headerType & (2)) != 0:
            normalMap = loadMatrix3D(f, headerDepthHeight, headerDepthWidth, 3, "f")

        confMap = np.zeros((0, 0), dtype = 'float')
        # confidence-map
        if (headerType & (4) != 0):
            confMap = loadMatrix3D(f, headerDepthHeight, headerDepthWidth, 3, "f")

    return imageFileName, IDs, [headerImgHeight, headerImgWidth], \
           K, R, C, headerdMin, headerdMax, \
           depthMap, normalMap, confMap


if __name__ == '__main__':
    fileName = "E:/dataset/mesh/workspace_d301/dense/dmap/depth0000.dmap"
    imageFileName, IDs, imgSize, K, R, C, dMin, dMax, depthMap, normalMap, confMap = ImportDepthDataRaw(fileName)





