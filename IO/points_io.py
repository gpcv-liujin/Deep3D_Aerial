#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2022, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu


import argparse
import os
import numpy as np
import time
import re
import sys
sys.path.append("..")

from plyfile import PlyData, PlyElement
from tools.utils import *


def save_points_to_ply(out_folder, ref_view_name, xyz, color, normal=None, skip_line=1):
    _, num = xyz.shape

    if num < 100:
        pass
    else:
        # for the final point cloud
        plyfilename = os.path.join(out_folder, '{}.ply'.format(ref_view_name))

        dir = os.path.dirname(plyfilename)
        if not os.path.exists(dir):
            os.mkdir(dir)

        vertexs = []
        vertex_colors = []
        vertex_normals = []

        vertexs.append(xyz.transpose((1, 0)))
        if normal is None:
            normal = np.zeros(np.shape(xyz), dtype=float)
        vertex_normals.append(normal.transpose((1, 0)))
        vertex_colors.append((color).astype(np.uint8))

        vertexs2 = np.concatenate(vertexs, axis=0)
        vertex_normals2 = np.concatenate(vertex_normals, axis=0)
        vertex_colors2 = np.concatenate(vertex_colors, axis=0)

        vertexs2 = np.array([tuple(v) for v in vertexs2[0::int(skip_line)]],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_normals2 = np.array([tuple(v) for v in vertex_normals2[0::int(skip_line)]],
                            dtype=[('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
        vertex_colors2 = np.array([tuple(v) for v in vertex_colors2[0::int(skip_line)]],
                                  dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        vertex_all = np.empty(len(vertexs2), vertexs2.dtype.descr + vertex_normals2.dtype.descr + vertex_colors2.dtype.descr)
        for prop in vertexs2.dtype.names:
            vertex_all[prop] = vertexs2[prop]
        for prop in vertex_normals2.dtype.names:
            vertex_all[prop] = vertex_normals2[prop]
        for prop in vertex_colors2.dtype.names:
            vertex_all[prop] = vertex_colors2[prop]

        el = PlyElement.describe(vertex_all, 'vertex')
        PlyData([el]).write(plyfilename)
        print("saving the final model to", plyfilename)


def save_points_to_bin(out_folder, ref_view_name, xyz, color):

    binfilename = os.path.join(out_folder, '{}.bin'.format(ref_view_name))
    dir = os.path.dirname(binfilename)
    if not os.path.exists(dir):
        os.mkdir(dir)

    _, num = xyz.shape

    if num < 100:
        pass
    else:
        xyz = xyz.transpose((1, 0))
        pc = np.hstack((xyz, color))
        pc.tofile(binfilename)
        print("saving the final model to", binfilename)


def read_points_from_bin(binfilename):
    pc = np.fromfile(binfilename, dtype=np.float64)
    pc = pc.reshape(-1, 6)

    return pc


def read_points_from_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    # vertexs = plydata['vertex'].data
    # x = vertexs['x']
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    r = plydata['vertex']['red']
    g = plydata['vertex']['green']
    b = plydata['vertex']['blue']
    # print(plydata['vertex'])

    pts = dict()
    pts["position"] = np.array([x, y, z]).T
    pts["color"] = np.array([r, g, b]).T
    pts["border"] = np.array([np.min(x), np.max(x),
                              np.min(y), np.max(y),
                              np.min(z), np.max(z)])

    return pts


class LasDataLoader(object):
    def __init__(self, root, batch_size, num_worker=0, shuffle=False, drop_last=True):
        self.root = root
        self.batch_size = batch_size
        self.las_paths = get_all_paths(root, ".las")
        self.len = len(self.las_paths)
        self.num_worker = num_worker
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.border = np.zeros(6)

        self.border_dir = os.path.join(self.root, "border").replace("\\", "/")
        self.border_path = os.path.join(self.border_dir, "point_cloud.border").replace("\\", "/")

        if not os.path.exists(self.border_dir):
            os.mkdir(self.border_dir)

        if shuffle:
            self.sort_shuffle()
        if os.path.exists(self.border_path):
            self.load_border_from_file(self.border_path)
        else:
            if self.num_worker > 0:
                self.get_all_border_multi_worker()
            else:
                self.get_all_border()

            self.get_border_of_all_point_cloud()
            self.save_border_as_file(self.border_path, self.border)

    @staticmethod
    def load_las_from_file(path):
        from laspy import file
        from laspy import header
        p = file.File(path, mode='r')
        # print("=====>Read Finished in {}s. Total num:{} !".format(end - start, p.__len__()))
        pts = dict()
        pts["position"] = np.array([p.x, p.y, p.z])
        pts["color"] = np.array([p.red / 256.0, p.green / 256.0, p.blue / 256.0])
        pts["border"] = np.array([p.header.min[0], p.header.max[0],
                                  p.header.min[1], p.header.max[1],
                                  p.header.min[2], p.header.max[2]])

        return pts

    @staticmethod
    def calculate_mer(borders):
        """
        borders:numpy.ndarray width shape (n, 6)
        a border: [x_min, x_max, y_min, y_max, z_min, z_max]
        Minimum external rectangle of the bounder
        """
        # print("border", borders)
        mer = np.zeros(6)
        for i in range(6):
            if i % 2:
                mer[i] = np.max(borders[:, i])
            else:
                mer[i] = np.min(borders[:, i])

        return mer

    @staticmethod
    def calculate_mir(borders):
        """
        borders:numpy.ndarray width shape (n, 6)
        a border: [x_min, x_max, y_min, y_max, z_min, z_max]
        Maximum inscribed rectangle of the bounders
        """
        mir = np.zeros(6)
        for i in range(6):
            if i % 2:
                mir[i] = np.min(borders[:, i])
            else:
                mir[i] = np.max(borders[:, i])

        return mir

    @staticmethod
    def batch_calculate_intersection_or_not(borders, bbx_border):
        n_num = borders.shape[0]
        mer = np.zeros((n_num, 4))
        bbx_border_repeat = np.expand_dims(bbx_border, 0).repeat(n_num, axis=0)

        update_flag0 = (borders[:, 0] > bbx_border_repeat[:, 0])
        update_flag2 = (borders[:, 2] > bbx_border_repeat[:, 2])

        update_flag1 = (borders[:, 1] < bbx_border_repeat[:, 1])
        update_flag3 = (borders[:, 3] < bbx_border_repeat[:, 3])

        mer[:, 0][update_flag0] = borders[:, 0][update_flag0]
        mer[:, 0][~update_flag0] = bbx_border_repeat[:, 0][~update_flag0]
        mer[:, 2][update_flag2] = borders[:, 2][update_flag2]
        mer[:, 2][~update_flag2] = bbx_border_repeat[:, 2][~update_flag2]

        mer[:, 1][update_flag1] = borders[:, 1][update_flag1]
        mer[:, 1][~update_flag1] = bbx_border_repeat[:, 1][~update_flag1]
        mer[:, 3][update_flag3] = borders[:, 3][update_flag3]
        mer[:, 3][~update_flag3] = bbx_border_repeat[:, 3][~update_flag3]

        det_x = mer[:, 1] - mer[:, 0]
        det_y = mer[:, 3] - mer[:, 2]

        is_intersected = np.logical_and((det_x > 0), (det_y > 0))

        return is_intersected

    @staticmethod
    def calculate_border_area(border):
        return (border[3] - border[2]) * (border[1] - border[0])

    def calculate_intersection(self, pt_border, bbx_border):
        border_stack = np.stack((pt_border, bbx_border), axis=0)
        intersection_border = self.calculate_mir(border_stack)
        mer_border = self.calculate_mer(border_stack)

        area_intersection_border = self.calculate_border_area(intersection_border)
        area_mer_border = self.calculate_border_area(mer_border)

        overlap = area_intersection_border / area_mer_border

        return intersection_border, overlap

    def get_item(self, idx):
        position = []
        color = []
        border = []

        start_id = idx * self.batch_size
        end_id = start_id + self.batch_size

        if self.drop_last:
            if end_id >= len(self.las_paths):
                return None
        else:
            end_id = len(self.las_paths) - 1

        for i in range(start_id, end_id):
            pts = self.load_las_from_file(self.las_paths[i])
            position.append(self.get_position(pts))
            color.append(self.get_color(pts))
            border.append(self.get_border(pts))

        return {"position": np.concatenate(position, axis=1),
                "color": np.concatenate(color, axis=1),
                "border": self.calculate_mer(np.stack(border, axis=0))}

    def batch_load_las_from_files(self, las_files, idx, batch_size):
        position = []
        color = []
        border = []

        if batch_size <= 0:
            batch_size = len(las_files) - 1
            idx = 0

        start_id = idx * batch_size
        end_id = start_id + batch_size

        if end_id > len(las_files) - 1:
            end_id = len(las_files) - 1

        for i in range(start_id, end_id):
            pts = self.load_las_from_file(las_files[i])
            position.append(self.get_position(pts))
            color.append(self.get_color(pts))
            border.append(self.get_border(pts))

        return {"position": np.concatenate(position, axis=1),
                "color": np.concatenate(color, axis=1),
                "border": self.calculate_mer(np.stack(border, axis=0))}

    def get_las_paths_in_border(self, bbx_border):
        from tqdm import tqdm
        print("Loading.... the point cloud in border: {}".format(bbx_border))

        border = np.zeros((len(self.las_paths), 6))
        idx = 0
        for lp in tqdm(self.las_paths):
            # Change 2021-12-21 !!!!!!!!!
            base_name = os.path.split(lp)[-1].split(".")[0] + ".txt"
            dir_name = os.path.basename(os.path.dirname(lp))
            border_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))

            pt_border = self.load_border_from_file(border_path)
            border[idx, :] = pt_border

            idx += 1

        is_intersection = self.batch_calculate_intersection_or_not(border, bbx_border)
        selected_las_paths = [self.las_paths[i] for i in range(len(self.las_paths)) if is_intersection[i]]
        print("{} files selected in total.".format(len(selected_las_paths)))

        return selected_las_paths

    def get_height_min_max_in_border(self, selected_las_paths):
        h_min, h_max = 99999, -99999
        for lp in selected_las_paths:
            # Change 2021-12-21 !!!!!!!!!
            base_name = os.path.split(lp)[-1].split(".")[0] + ".txt"
            dir_name = os.path.basename(os.path.dirname(lp))
            border_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))

            pt_border = self.load_border_from_file(border_path)
            if h_min > pt_border[4]:
                h_min = pt_border[4]
            if h_max < pt_border[5]:
                h_max = pt_border[5]

        return h_min, h_max

    def get_mer_of_all_selected_las(self, selected_las_paths):
        borders = []
        for lp in selected_las_paths:
            base_name = os.path.split(lp)[-1].split(".")[0] + ".txt"
            dir_name = os.path.basename(os.path.dirname(lp))
            border_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))

            borders.append(self.load_border_from_file(border_path))

        borders = np.array(borders)

        return self.calculate_mer(borders)

    @staticmethod
    def get_position(pts):
        return pts["position"]

    @staticmethod
    def get_color(pts):
        return pts["color"]

    @staticmethod
    def get_border(pts):
        return pts["border"]

    def sort_shuffle(self):
        import random
        random.shuffle(self.las_paths)

    def get_all_border(self):
        from tqdm import tqdm
        print("calculating and saving the border...")
        for i in tqdm(range(len(self.las_paths))):
            pts = self.load_las_from_file(self.las_paths[i])
            border = self.get_border(pts)

            base_name = os.path.split(self.las_paths[i])[-1].split(".")[0] + ".txt"
            dir_name = os.path.basename(os.path.dirname(self.las_paths[i]))
            save_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))

            self.save_border_as_file(save_path, border)

    def worker(self, i):
        print("======> saving {}/{}========>".format(i, len(self.las_paths)))
        print(self.las_paths[i])
        pts = self.load_las_from_file(self.las_paths[i])
        border = self.get_border(pts)

        base_name = os.path.split(self.las_paths[i])[-1].split(".")[0] + ".txt"
        dir_name = os.path.basename(os.path.dirname(self.las_paths[i]))
        save_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))

        self.save_border_as_file(save_path, border)

    def get_all_border_multi_worker(self):
        print("calculating and saving the border...")
        import multiprocessing

        pool = multiprocessing.Pool(self.num_worker)
        for i in range(len(self.las_paths)):
            pool.apply_async(self.worker, {i})
        pool.close()
        pool.join()

    def get_border_of_all_point_cloud(self):
        from tqdm import tqdm
        border_paths = get_all_paths(self.border_dir, '.txt')
        borders = np.zeros((len(self.las_paths), 6))

        idx = 0
        for bp in tqdm(border_paths):
            if os.path.splitext(bp)[-1] == ".txt":
                bp_path = os.path.join(self.border_dir, bp).replace("\\", "/")
                borders[idx, :] = (self.load_border_from_file(bp_path))
                idx += 1

        self.border = self.calculate_mer(borders)

    @staticmethod
    def save_border_as_file(fname, border):
        text = ""
        for b in border:
            text += str(b) + "\n"

        with open(fname, "w") as f:
            f.write(text)

    @staticmethod
    def load_border_from_file(border_path):
        with open(border_path, "r") as f:
            text = f.read().splitlines()

        return np.array(text[0:6], dtype=np.float)

    @staticmethod
    def show_las(pts):
        import matplotlib.pyplot as plt

        fig = plt.figure(dpi=500)
        ax = fig.add_subplot(111, projection='3d')

        colors = pts["color"] / 255

        points = pts["position"]
        ax.scatter(pts[:, 0], points[:, 1], points[:, 2], cmap='spectral', c=colors,
                   s=0.5, linewidth=0, alpha=1, marker=".")

        plt.title('Point Cloud')
        ax.axis('scaled')  # {equal, scaled}
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


class plyDataLoader(object):
    def __init__(self, root, batch_size, num_worker=0, shuffle=False, drop_last=True):
        self.root = root
        self.batch_size = batch_size
        self.ply_paths = get_all_paths(root, ".ply")
        self.len = len(self.ply_paths)
        self.num_worker = num_worker
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.border = np.zeros(6)

        self.border_dir = os.path.join(self.root, "border").replace("\\", "/")
        self.border_path = os.path.join(self.border_dir, "point_cloud.border").replace("\\", "/")
        if not os.path.exists(self.border_dir):
            os.mkdir(self.border_dir)

        if shuffle:
            self.sort_shuffle()
        if os.path.exists(self.border_path):
            self.load_border_from_file(self.border_path)
        else:
            if num_worker > 0:
                self.get_all_border_multi_worker()
            else:
                self.get_all_border()
            self.get_border_of_all_point_cloud()
            self.save_border_as_file(self.border_path, self.border)

    @staticmethod
    def load_ply_from_file(path):
        """ read XYZ point cloud from filename PLY file """
        plydata = PlyData.read(path)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        r = plydata['vertex']['red']
        g = plydata['vertex']['green']
        b = plydata['vertex']['blue']

        pts = dict()
        pts["position"] = np.array([x, y, z])
        pts["color"] = np.array([r, g, b])
        pts["border"] = np.array([np.min(x), np.max(x),
                                  np.min(y), np.max(y),
                                  np.min(z), np.max(z)])

        return pts

    @staticmethod
    def calculate_mer(borders):
        """
        borders:numpy.ndarray width shape (n, 6)
        a border: [x_min, x_max, y_min, y_max, z_min, z_max]
        Minimum external rectangle of the bounder
        """
        mer = np.zeros(6)
        for i in range(6):
            if i % 2:
                mer[i] = np.max(borders[:, i])
            else:
                mer[i] = np.min(borders[:, i])

        return mer

    @staticmethod
    def calculate_mir(borders):
        """
        borders:numpy.ndarray width shape (n, 6)
        a border: [x_min, x_max, y_min, y_max, z_min, z_max]
        Maximum inscribed rectangle of the bounders
        """
        mir = np.zeros(6)
        for i in range(6):
            if i % 2:
                mir[i] = np.min(borders[:, i])
            else:
                mir[i] = np.max(borders[:, i])

        return mir

    @staticmethod
    def batch_calculate_intersection_or_not(borders, bbx_border):
        """
        :param borders: (n, 6)
        :param bbx_border: (6)
        :return:overlap (n, 6)
        """
        n_num = borders.shape[0]
        mer = np.zeros((n_num, 4))
        bbx_border_repeat = np.expand_dims(bbx_border, 0).repeat(n_num, axis=0)

        update_flag0 = (borders[:, 0] > bbx_border_repeat[:, 0])
        update_flag2 = (borders[:, 2] > bbx_border_repeat[:, 2])

        update_flag1 = (borders[:, 1] < bbx_border_repeat[:, 1])
        update_flag3 = (borders[:, 3] < bbx_border_repeat[:, 3])

        mer[:, 0][update_flag0] = borders[:, 0][update_flag0]
        mer[:, 0][~update_flag0] = bbx_border_repeat[:, 0][~update_flag0]
        mer[:, 2][update_flag2] = borders[:, 2][update_flag2]
        mer[:, 2][~update_flag2] = bbx_border_repeat[:, 2][~update_flag2]

        mer[:, 1][update_flag1] = borders[:, 1][update_flag1]
        mer[:, 1][~update_flag1] = bbx_border_repeat[:, 1][~update_flag1]
        mer[:, 3][update_flag3] = borders[:, 3][update_flag3]
        mer[:, 3][~update_flag3] = bbx_border_repeat[:, 3][~update_flag3]

        det_x = mer[:, 1] - mer[:, 0]
        det_y = mer[:, 3] - mer[:, 2]

        is_intersected = np.logical_and((det_x > 0), (det_y > 0))

        return is_intersected

    @staticmethod
    def calculate_border_area(border):
        """
        borders:numpy.ndarray width shape (n, 6)
        a border: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        return (border[3] - border[2]) * (border[1] - border[0])

    def calculate_intersection(self, pt_border, bbx_border):
        """
        pt_border: (n, 6)
        bbx_border: (6)
        """
        border_stack = np.stack((pt_border, bbx_border), axis=0)
        intersection_border = self.calculate_mir(border_stack)
        mer_border = self.calculate_mer(border_stack)

        area_intersection_border = self.calculate_border_area(intersection_border)
        area_mer_border = self.calculate_border_area(mer_border)

        overlap = area_intersection_border / area_mer_border

        return intersection_border, overlap

    def get_item(self, idx):
        position = []
        color = []
        border = []

        start_id = idx * self.batch_size
        end_id = start_id + self.batch_size

        if self.drop_last:
            if end_id >= len(self.ply_paths):
                return None
        else:
            end_id = len(self.ply_paths) - 1

        for i in range(start_id, end_id):
            pts = self.load_ply_from_file(self.ply_paths[i])
            position.append(self.get_position(pts))
            color.append(self.get_color(pts))
            border.append(self.get_border(pts))

        return {"position": np.concatenate(position, axis=1),
                "color": np.concatenate(color, axis=1),
                "border": self.calculate_mer(np.stack(border, axis=0))}

    def batch_load_ply_from_files(self, ply_files, idx, batch_size):
        position = []
        color = []
        border = []

        if batch_size <= 0:
            batch_size = len(ply_files) - 1
            idx = 0

        start_id = idx * batch_size
        end_id = start_id + batch_size

        if end_id > len(ply_files) - 1:
            end_id = len(ply_files) - 1

        for i in range(start_id, end_id):
            pts = self.load_ply_from_file(ply_files[i])
            position.append(self.get_position(pts))
            color.append(self.get_color(pts))
            border.append(self.get_border(pts))

        return {"position": np.concatenate(position, axis=1),
                "color": np.concatenate(color, axis=1),
                "border": self.calculate_mer(np.stack(border, axis=0))}

    def get_ply_paths_in_border(self, bbx_border):
        from tqdm import tqdm
        border = np.zeros((len(self.ply_paths), 6))

        idx = 0
        for lp in self.ply_paths:
            base_name = os.path.split(lp)[-1].split(".")[0] + ".txt"
            dir_name = os.path.basename(os.path.dirname(lp))
            border_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))

            pt_border = self.load_border_from_file(border_path)
            border[idx, :] = pt_border
            idx += 1

        is_intersection = self.batch_calculate_intersection_or_not(border, bbx_border)
        selected_ply_paths = [self.ply_paths[i] for i in range(len(self.ply_paths)) if is_intersection[i]]
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print("{} files selected in range {}.".format(len(selected_ply_paths), bbx_border))

        return selected_ply_paths

    def get_all_ply_paths(self):
        print("Loading the point cloud in file: {}......".format(self.root))

        border = np.zeros((len(self.ply_paths), 6))
        idx = 0

        for lp in self.ply_paths:
            base_name = os.path.split(lp)[-1].split(".")[0] + ".txt"
            dir_name = os.path.basename(os.path.dirname(lp))
            border_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))

            pt_border = self.load_border_from_file(border_path)
            border[idx, :] = pt_border
            idx += 1

        all_ply_paths = [self.ply_paths[i] for i in range(len(self.ply_paths))]
        print("find {} point cloud files in total.".format(len(all_ply_paths)))

        return all_ply_paths

    def get_height_min_max_in_border(self, selected_ply_paths):
        h_min, h_max = 99999, -99999
        for lp in selected_ply_paths:
            base_name = os.path.split(lp)[-1].split(".")[0] + ".txt"
            dir_name = os.path.basename(os.path.dirname(lp))
            border_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))

            pt_border = self.load_border_from_file(border_path)
            if h_min > pt_border[4]:
                h_min = pt_border[4]
            if h_max < pt_border[5]:
                h_max = pt_border[5]

        return h_min, h_max

    def get_mer_of_all_selected_ply(self, selected_ply_paths):
        borders = []
        for lp in selected_ply_paths:
            base_name = os.path.split(lp)[-1].split(".")[0] + ".txt"
            dir_name = os.path.basename(os.path.dirname(lp))
            border_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))

            borders.append(self.load_border_from_file(border_path))

        borders = np.array(borders)

        return self.calculate_mer(borders)

    @staticmethod
    def get_position(pts):
        return pts["position"]

    @staticmethod
    def get_color(pts):
        return pts["color"]

    @staticmethod
    def get_border(pts):
        return pts["border"]

    def sort_shuffle(self):
        import random
        random.shuffle(self.ply_paths)

    def get_all_border(self):
        from tqdm import tqdm
        print("calculating and saving the border...")
        for i in tqdm(range(len(self.ply_paths))):
            pts = self.load_ply_from_file(self.ply_paths[i])
            border = self.get_border(pts)

            base_name = os.path.split(self.ply_paths[i])[-1].split(".")[0] + ".txt"
            dir_name = os.path.basename(os.path.dirname(self.ply_paths[i]))
            save_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))

            self.save_border_as_file(save_path, border)

    def worker(self, i):
        print("======> saving {}/{}========>".format(i, len(self.ply_paths)))
        pts = self.load_ply_from_file(self.ply_paths[i])
        border = self.get_border(pts)

        base_name = os.path.split(self.ply_paths[i])[-1].split(".")[0] + ".txt"
        dir_name = os.path.basename(os.path.dirname(self.ply_paths[i]))
        save_path = os.path.join(self.border_dir, "{}/{}".format(dir_name, base_name))
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        self.save_border_as_file(save_path, border)

    def get_all_border_multi_worker(self):
        print("calculating and saving the border...")
        import multiprocessing

        pool = multiprocessing.Pool(self.num_worker)
        for i in range(len(self.ply_paths)):
            pool.apply_async(self.worker, {i})
        pool.close()
        pool.join()

    def get_border_of_all_point_cloud(self):
        border_paths = get_all_paths(self.border_dir, '.txt')
        borders = np.zeros((len(self.ply_paths), 6))

        idx = 0
        for bp in border_paths:
            if os.path.splitext(bp)[-1] == ".txt":
                bp_path = os.path.join(self.border_dir, bp).replace("\\", "/")
                borders[idx, :] = (self.load_border_from_file(bp_path))
                idx += 1

        self.border = self.calculate_mer(borders)

    @staticmethod
    def save_border_as_file(fname, border):
        text = ""
        for b in border:
            text += str(b) + "\n"

        with open(fname, "w") as f:
            f.write(text)

    @staticmethod
    def load_border_from_file(border_path):
        with open(border_path, "r") as f:
            text = f.read().splitlines()

        return np.array(text[0:6], dtype=np.float)

    @staticmethod
    def show_ply(pts):
        import matplotlib.pyplot as plt

        fig = plt.figure(dpi=500)
        ax = fig.add_subplot(111, projection='3d')

        colors = pts["color"] / 255

        points = pts["position"]
        ax.scatter(pts[:, 0], points[:, 1], points[:, 2], cmap='spectral', c=colors,
                   s=0.5, linewidth=0, alpha=1, marker=".")

        plt.title('Point Cloud')
        ax.axis('scaled')  # {equal, scaled}
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
