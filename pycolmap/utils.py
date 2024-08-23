# Copyright (c) 2022, Wuhan University and GPCV.
# All rights reserved.
# Author: Jin Liu


import os
import sys
import numpy as np
import sqlite3


def join(path, item):
    return os.path.join(path, item).replace("\\", "/")

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def matches_as_array(database):
    # return points dict{key: pair_id, value: [id, x1, y1, x2, y2], [id, x1, y1, x2, y2]...}
    conn = sqlite3.connect(database)
    c = conn.cursor()

    sql_seq = """SELECT pair_id, cols, data FROM matches"""

    matches = dict()
    for image_id, cols, data in c.execute(sql_seq):
        if data != None:
            matches[image_id] = blob_to_array(data, np.uint32, (-1, cols))[:, :2]

    return matches


def write_pair_txt(txt_file, score):
    text = "{}\n".format(len(score))

    for pair in score:
        text += "{}\n{} ".format(pair[0], len(pair[1]))
        for s in pair[1]:
            text += "{} {:.4f} ".format(s[0], s[1])
        text += "\n"

    with open(txt_file, "w") as f:
        f.write(text)

