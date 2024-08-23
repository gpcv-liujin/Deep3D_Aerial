# Copyright (c) 2022, Wuhan University and GPCV.
# All rights reserved.
# Author: Jin Liu and Jian Gao


import os
import numpy as np
import math
from pycolmap.read_write_model import read_model
from pycolmap.utils import image_ids_to_pair_id, pair_id_to_image_ids, join, matches_as_array


def calculate_block_border(sparse_path, base_size=None, base_overlap=1, bbx_border=None):
    """
    :base_size = [x_size, y_size, z_size]
    :param bbx_border:  a border: [x_min, x_max, y_min, y_max, z_min, z_max]
    return: blocks_list [x_min, x_max, y_min, y_max, z_min, z_max] *n
            scene_block [x_min, x_max, y_min, y_max, z_min, z_max]
    """

    # Read and write model
    cams, images, point3ds = read_model(sparse_path)

    total_xyz = []
    for pt in point3ds.keys():
        xyz = point3ds[pt].xyz.tolist()
        total_xyz.append(xyz)
    total_xyz = np.array(total_xyz)

    min_x, max_x = np.percentile(total_xyz[:, 0], [0.5, 99.5])
    min_y, max_y = np.percentile(total_xyz[:, 1], [0.5, 99.5])
    min_z, max_z = np.percentile(total_xyz[:, 2], [0.5, 99.5])
    print("------------scene range: ----------------")
    print("min: [{}, {}, {}]".format(min_x, min_y, min_z))
    print("max: [{}, {}, {}]".format(max_x, max_y, max_z))

    if bbx_border is not None:
        scene_border = bbx_border
    else:
        scene_border = [min_x, max_x, min_y, max_y, min_z, max_z]

    if base_size is not None:
        scene_size = [float(x) for x in base_size]
    else:
        x_size = (max_x - min_x) / 2.0
        y_size = (max_y - min_y) / 2.0
        z_size = (max_z - min_z) / 1.0
        scene_size = [x_size, y_size, z_size]

    overlap_size = [base_overlap, base_overlap, base_overlap]  # units: m

    blocks_list = []
    x_block_num = math.ceil((scene_border[1] - scene_border[0]) / scene_size[0])
    y_block_num = math.ceil((scene_border[3] - scene_border[2]) / scene_size[1])

    print("=======> total: {}*{} blocks.".format(y_block_num, x_block_num))
    for j in range(y_block_num):
        for i in range(x_block_num):
            block_range = [0, 0, 0, 0, min_z, max_z]
            block_range[0] = scene_border[0] + i * scene_size[0] - overlap_size[0]
            block_range[1] = block_range[0] + scene_size[0] + overlap_size[0]
            block_range[2] = scene_border[2] + j * scene_size[1] - overlap_size[1]
            block_range[3] = block_range[2] + scene_size[1] + overlap_size[1]
            blocks_list.append(block_range)

    return blocks_list, scene_border


def select_all_in_range_as_reference_images(model_path, range):
    cameras, images, points3d = read_model(model_path)

    image_ids_list = []
    points_list = list(points3d.keys())

    for points_id in points_list:
        if points_id > 0:
            xyz = points3d[points_id].xyz
            if range[0] < xyz[0] < range[1] and range[2] < xyz[1] < range[3]:
                image_ids = points3d[points_id].image_ids
                image_ids_list.extend(image_ids)

    ref_image_set = list(set(image_ids_list))

    return ref_image_set, cameras, images, points3d


def calculate_src_images_score_based_tie_points_num(reference_images_list, matches_array):

    tie_points_num = [[] for ref in reference_images_list]
    score_total = [0 for ref in reference_images_list]

    for pair_id, match_pts in matches_array.items():
        img_id1, img_id2 = pair_id_to_image_ids(pair_id)
        img_id1 = int(img_id1)
        img_id2 = int(img_id2)

        if img_id1 in reference_images_list:
            idx = reference_images_list.index(img_id1)
            tie_points_num[idx].append([img_id2, match_pts.shape[0]])
            score_total[idx] += match_pts.shape[0]

        if img_id2 in reference_images_list:
            idx = reference_images_list.index(img_id2)
            tie_points_num[idx].append([img_id1, match_pts.shape[0]])
            score_total[idx] += match_pts.shape[0]

    score = []
    for i in range(len(reference_images_list)):
        tie_points_num[i] = [[item[0], item[1]/score_total[i]] for item in tie_points_num[i]]
        tie_points_num[i].sort(key=lambda x: x[1], reverse=True)

        if len(tie_points_num[i]) > 2: # Ensure MVS
            score.append([reference_images_list[i], tie_points_num[i]])

    return score


def calculate_src_images_score_based_triangulated_points(reference_images_list, images, points3D):
    score = []

    # select all as reference images
    # cameras, images, points3d = read_model(model_path)
    # reference_images_list = list(images.keys())

    for ref_image_id in reference_images_list:
        src_image_ids_list = []
        view_point3D_ids = images[ref_image_id].point3D_ids

        for ids in view_point3D_ids:
            if ids > 0:
                src_image_ids = points3D[ids].image_ids
                src_image_ids_list.extend(src_image_ids)
        count_set = list(set(src_image_ids_list))  # matching view number

        if len(count_set) > 3:   # Ensure MVS
            count_set.remove(ref_image_id)
            tie_points_num = [[item, src_image_ids_list.count(item)] for item in count_set]
            tie_points_num.sort(key=lambda x: x[1], reverse=True)
            max_tie_points_num = tie_points_num[0][1]
            valid_tie_points_num = [x for x in tie_points_num if x[1] > 10 and x[1] > max_tie_points_num/10.0]
            score.append([ref_image_id, valid_tie_points_num])

    return score


def select_view_based_viewed_points(database_path, sparse_path, range, mode='triangulated_points'):
    """
    database_path:
    sparse_path:
    pair_path:
    mode: 'triangulated_points' or 'tie_points'
    range: list [xmin, xmax, ymin, ymax, zmin, zmax] * n
    """

    block_num = len(range)
    print(range)
    print("------------ Divide all selected views into {} blocks. -------------".format(block_num))

    total_viewpair = []
    total_ref_view_in_range = []

    for sub_range in range:
        reference_images_list, cameras, images, points3d = select_all_in_range_as_reference_images(sparse_path, sub_range)

        if mode == 'triangulated_points':
            score = calculate_src_images_score_based_triangulated_points(reference_images_list, images, points3d)

        elif mode == 'tie_points':
            if os.path.exists(database_path):
                matches = matches_as_array(database_path)
                score = calculate_src_images_score_based_tie_points_num(reference_images_list, matches)
            else:
                raise Exception("{} does not exist!".format(mode))

        else:
            raise Exception("{}? Not implemented yet!".format(mode))

        if len(score) > 0:
            total_ref_view_in_range.append([sub_range, [s[0] for s in score]])

        for s in score:
            if s not in total_viewpair:
                total_viewpair.append(s)


    return total_ref_view_in_range, total_viewpair













