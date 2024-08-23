#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2022, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu

import shutil
import sys
import time
sys.path.append("..")
import warnings
import matplotlib.pyplot as plt
import collections
import pandas as pd
import struct
import math
import argparse
from PIL import Image
from IO.pfm import *
from IO.params_io import save_border_as_file
from IO.mvs_io import Vertex, Interface
from fuse.consistency_check_n import ConsistencyChecker
from tools.utils import *
import cupy as cp


# this cord except input Tcw [Rcw| tcw], and K[3*3],
# keep cams_ori=XrightYdown and images_ori=Tcw for CasRednet and rednet results.

parser = argparse.ArgumentParser(description='filter and fuse.')
parser.add_argument('--project_folder', type=str, default=r'F:\pipeline\pipeline_test\workspace_virtual_1')
parser.add_argument('--image_path', type=str, default=r'X:/liujin_densematching/MVS_traindata/Munchen_testlargeImages/split_test/MVS')
parser.add_argument('--sparse_path', type=str, default=r'X:/liujin_densematching/MVS_traindata/Munchen_testlargeImages/split_test/MVS')
parser.add_argument('--mvs_path', type=str, default=r'X:/liujin_densematching/MVS_traindata/Munchen_testlargeImages/split_test/MVS')
parser.add_argument('--output_path', type=str, default=r'X:/liujin_densematching/MVS_traindata/Munchen_testlargeImages/split_test/Fusion')

parser.add_argument('--min_geo_consist_num', default=4, help='geo_consist_num')
parser.add_argument('--photomatric_threshold', default=0.2)
parser.add_argument('--fusion_num', default=10, help='use how many depths to fuse')
parser.add_argument('--position_threshold', default=1, help='Maximum relative difference between measured and projected pixel')
parser.add_argument('--depth_threshold', default=0.01, help='Maximum relative difference between measured and projected depth')
parser.add_argument('--normal_threshold', default=10, help='Maximum angular difference in degrees of normals of pixels to be fused')
parser.add_argument('--skip_line', default=2, help='skip_line')

parser.add_argument('--camera_scale', default=1, help='camera_scale')
parser.add_argument('--cams_ori', default='XrightYdown', help='Camera orientation, 0 XrightYdown; 1 XrightYup')
parser.add_argument('--images_ori', default='Tcw', help='image orientation, 0 Tcw; 1 Twc; 2 [Rcw,twc]')
parser.add_argument('--implement', default='cupy', help="numpy, cupy or torch")
parser.add_argument('--save_mask_or_not', type=bool, default=True, help="save mask or not")
parser.add_argument('--pc_format', type=str, default="ply", help="save point cloud as bin/ply")

args = parser.parse_args()


class Fuse_Depth_Map:
    def __init__(self, project_folder, sparse_path, mvs_path, output_path, fusion_num=30,  min_geometric_consistency_num=3, photometric_threshold=0.2,
                 position_consistency_threshold=1.0, depth_consistency_threshold=0.01, normal_consistency_threshold=90.0, camera_scale=1.0, cams_ori='XrightYdown',
                 images_ori='Tcw', implement='cupy', save_mask_or_not=True, save_temp=True, pc_format='ply'):

        self.project_folder = project_folder.replace("\\", "/")
        self.image_path = []
        self.sparse_path = sparse_path
        self.depth_path = mvs_path
        self.output_path = output_path
        self.tmp_path = self.output_path + "/tmp"
        self.mask_path = self.output_path + "/mask"

        # parameters
        self.fusion_num = fusion_num
        self.min_geo_consist_num = min_geometric_consistency_num
        self.photometric_threshold = photometric_threshold
        self.position_consistency_threshold = position_consistency_threshold
        self.depth_consistency_threshold = depth_consistency_threshold
        self.normal_consistency_threshold = normal_consistency_threshold

        self.skip_line = 2
        self.camera_scale = camera_scale
        self.cams_ori = cams_ori
        self.images_ori = images_ori
        self.implement = implement
        self.save_mask_or_not = save_mask_or_not
        self.save_temp = save_temp
        self.pc_format = pc_format

        self.geometric_checker = ConsistencyChecker(
            self.position_consistency_threshold, self.depth_consistency_threshold, self.normal_consistency_threshold,
            self.photometric_threshold, self.implement)

        self.viewpair_path = self.project_folder + '/viewpair.txt'
        self.view_name_path = self.project_folder + '/image_path.txt'
        self.scene_block_path = self.project_folder + '/blocks.txt'

        self.sparse_camera_path = self.sparse_path + '/cameras.bin'
        self.sparse_image_path = self.sparse_path + '/images.bin'
        self.sparse_point_path = self.sparse_path + '/points3D.bin'
        self.map_image_dict = self.read_ImageID(self.sparse_image_path)

        self.view_name_list = self.read_name_list(self.view_name_path)
        self.pair_list = self.read_view_pair_text(self.viewpair_path, self.view_name_list, self.fusion_num)
        self.scene_block_list = self.read_scene_block_text(self.scene_block_path, self.pair_list)

        self.create_folder()

    def create_folder(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.exists(self.mask_path):
            os.mkdir(self.mask_path)
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)

    def read_camera_parameters(self, filename, scale=1.0):
        # read intrinsics and extrinsics
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        # extrinsics: line [1,5), 4x4 matrix  XrightYdown Tcw
        extr = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        extrinsics = self.create_extrinsics_matrix(extr, self.cams_ori, self.images_ori)

        # intrinsics: line [7-10), 3x3 matrix  K
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] *= scale

        # depth: line [11-12), 1x4 matrix  [depth_min, depth_interval, depth_num, depth_max]
        depth_range = np.fromstring(' '.join(lines[11:12]), dtype=np.float32, sep=' ').reshape((1, 4))

        # image information: line [13-14) 1x5 [width, height, image_id, image_name, image_path]
        # image_info = np.fromstring(' '.join(lines[13:14]), dtype=str, sep=' ')
        image_info = lines[13:14][0].split(' ')
        image_path = image_info[4]
        return intrinsics, extrinsics, image_path

    @ staticmethod
    def create_extrinsics_matrix(extrinsics, cams_ori, images_ori):
        """
        this cord except for Tcw[Rcw|tcw]
        """
        if cams_ori == 'XrightYup':
            O = np.eye(3, dtype=np.float32)
            O[1, 1] = -1
            O[2, 2] = -1
        elif cams_ori == 'XrightYdown':
            O = np.eye(3, dtype=np.float32)
        else:
            O = np.eye(3, dtype=np.float32)
            Exception("{} is not defined!".format(cams_ori))

        if images_ori == "Twc":
            R = extrinsics[0:3, 0:3]
            R2 = np.matmul(R, O)
            extrinsics[0:3, 0:3] = R2
            extrinsics = np.linalg.inv(extrinsics)  # convert to Tcw

        elif images_ori == "Rcw":
            R = extrinsics[0:3, 0:3]  # Rcw|twc
            R2 = np.matmul(O, R)
            R3 = np.linalg.inv(R2) 
            extrinsics[0:3, 0:3] = R3
            extrinsics = np.linalg.inv(extrinsics)  # convert to Tcw

        elif images_ori == "Tcw":
            extrinsics = np.linalg.inv(extrinsics)
            R = extrinsics[0:3, 0:3]
            R2 = np.matmul(R, O)
            extrinsics[0:3, 0:3] = R2
            extrinsics = np.linalg.inv(extrinsics)  # convert to Tcw
        else:
            Exception("{} is not defined!".format(images_ori))

        return extrinsics

    def read_img(self, filename, image_scale=1):
        img = Image.open(filename)

        # scale image
        if image_scale != 1:
            width, height = img.size
            nw = int(width * image_scale)
            nh = int(height * image_scale)
            img = img.resize((nw, nh), Image.ANTIALIAS)

        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img[:, :, 0:3]

    def read_mask(self, filename):
        return self.read_img(filename) > 0.5

    def read_normal(self, filename):
        normal_est = read_pfm(filename)[0]
        normal_est = normal_est * 2.0 - 1.0

        return normal_est


    @ staticmethod
    def save_mask(filename, mask):
        # save a binary mask
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.mkdir(dir)

        assert mask.dtype == np.bool
        mask = mask.astype(np.uint8) * 255
        Image.fromarray(mask).save(filename)

    @staticmethod
    def read_name_list(path):
        paths_list = {}
        names_list = {}
        cluster_list = open(path).read().split()
        total_num = int(cluster_list[0])

        for i in range(total_num):
            index = int(cluster_list[i * 3 + 1])  # index
            name = cluster_list[i * 3 + 2]
            p = cluster_list[i * 3 + 3]

            paths_list[index] = p
            names_list[index] = name

        return names_list

    @staticmethod
    def read_view_pair_text(pair_path, pair_name_list, view_num):
        metas_list = {}
        # read the pair file
        with open(pair_path) as f:
            num_viewpoint = int(f.readline())
            print("====>total {} ref_views.".format(num_viewpoint))
            # viewpoints
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                ref_view_name = os.path.splitext(pair_name_list[ref_view])[0]
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                src_views_name = [os.path.splitext(pair_name_list[src_view])[0] for src_view in src_views]

                # filter by no src view and fill to nviews
                if len(src_views_name) > 0:
                    if len(src_views_name) < view_num:
                        print("{} num_src_views:{} < num_views:{}".format(ref_view_name, len(src_views_name), view_num))
                        src_views_name += [src_views_name[0]] * (view_num - len(src_views_name))

                    metas_list[ref_view] = {"ref": ref_view_name,
                                            "src": src_views_name}

        return metas_list

    @staticmethod
    def read_scene_block_text(scene_block_path, metas_list):

        metas = []
        # read the scene range file
        with open(scene_block_path) as f:
            num_scene = int(f.readline())
            print("====>total {} scene blocks.".format(num_scene))
            for scene in range(num_scene):
                metas_in_range = {}
                scene_range = [float(x) for x in f.readline().rstrip().split()]
                ref_list_in_range = [int(x) for x in f.readline().rstrip().split()]

                view_list = []
                for i in ref_list_in_range:
                    views = metas_list[i]
                    view_list.append(views)
                metas_in_range["scene_range"] = scene_range
                metas_in_range["view_list"] = view_list
                metas.append(metas_in_range)

        return metas

    def read_ImageID(self, sparse_image_bin_path):

        def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
            data = fid.read(num_bytes)
            return struct.unpack(endian_character + format_char_sequence, data)

        mapImages = {}

        with open(sparse_image_bin_path, "rb") as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            cnt = 1
            for _ in range(num_reg_images):
                binary_image_properties = read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi")
                image_id = binary_image_properties[0] - 1
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
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
                image_name = os.path.splitext(image_name)[0]
                mapImages[image_name] = cnt
                cnt = cnt + 1

        return mapImages

    def TransferImageID(self, mapImages, visible_image_idxs):
        image_ids_transfer = []
        for image_id in visible_image_idxs:
            if str(image_id) in mapImages.keys():
                image_ids_transfer.append(mapImages[str(image_id)])
        return np.array(image_ids_transfer)

    def filter_depths(self):
        for view_key in self.pair_list.keys():
            view_pair = self.pair_list[view_key]
            ref_view = view_pair["ref"]
            src_views = view_pair["src"]

            ref_depth_est_path = join(self.depth_path, '{}_init.pfm'.format(ref_view))
            if os.path.exists(join(self.depth_path, '{}_init.pfm'.format(ref_view))):
                ref_depth_est_path = join(self.depth_path, '{}_init.pfm'.format(ref_view))
            if not os.path.exists(ref_depth_est_path):
                warnings.warn("{} not exists".format(ref_depth_est_path))
                continue

            start = time.time()

            # load the reference camera parameters
            ref_intrinsics, ref_extrinsics, image_path = self.read_camera_parameters(
                join(self.depth_path, '{}.txt'.format(ref_view)), self.camera_scale)
            self.image_path = os.path.dirname(os.path.dirname(image_path))
            # load the reference image
            # ref_img = self.read_img(join(self.image_path, '{}.jpg'.format(ref_view)))
            # load the estimated depth of the reference view
            ref_depth_est = read_pfm(ref_depth_est_path)[0]
            # load the photometric mask of the reference view
            confidence = read_pfm(join(self.depth_path, '{}_prob.pfm'.format(ref_view)))[0]
            photo_mask = confidence > self.photometric_threshold
            t1 = time.time()

            all_srcview_depth_ests = []
            geo_mask_sum = 0

            for src_view in src_views[:self.fusion_num]:
                src_depth_est_path = join(self.depth_path, '{}_init.pfm'.format(src_view))
                if not os.path.exists(join(self.depth_path, '{}_init.pfm'.format(src_view))):
                    print(join(self.depth_path, '{}_init.pfm'.format(src_view)))
                    warnings.warn("{} not exists".format(
                        join(self.depth_path, '{}_init.pfm'.format(src_view))))
                    continue

                # camera parameters of the source view
                src_intrinsics, src_extrinsics, _ = self.read_camera_parameters(
                    join(self.depth_path, '{}.txt'.format(src_view)), self.camera_scale)
                # the estimated depth of the source view
                src_depth_est = read_pfm(src_depth_est_path)[0]

                geo_mask, depth_reprojected, _, _ = self.geometric_checker.check(
                    ref_depth_est, ref_intrinsics, ref_extrinsics, src_depth_est,
                    src_intrinsics, src_extrinsics, confidence)

                geo_mask_sum += geo_mask.astype(np.int32)
                all_srcview_depth_ests.append(depth_reprojected)

                del geo_mask, depth_reprojected

            depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
            depth_est_averaged = depth_est_averaged.astype(np.float32)
            # at least N source views matched
            final_mask = np.array(geo_mask_sum >= self.min_geo_consist_num)

            t2 = time.time()

            if self.save_mask_or_not:
                self.save_mask(os.path.join(self.mask_path, "{}_photo.png".format(ref_view)), cp.asnumpy(photo_mask))
                self.save_mask(os.path.join(self.mask_path, "{}_final.png".format(ref_view)), cp.asnumpy(final_mask))
            print("ref-view {}, photo/final-mask:{}/{}".format(ref_view, photo_mask.mean(), final_mask.mean()))

            tmp_ref_depth_path = join(self.tmp_path, '{}_init.pfm'.format(ref_view))
            tmp_dir = os.path.dirname(tmp_ref_depth_path)
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            depth_est_averaged[~final_mask] = 0
            write_pfm(tmp_ref_depth_path, depth_est_averaged)

            t3 = time.time()
            print("filter time:{}; total time:{}".format((t2 - start), (t3 - start)))

    def fuse_depths(self, scene_range, view_list, scene_name):
        """
        # for each reference view and the corresponding source views
        # view_list = scene_block["view_list"]  #  a list ["ref", "src"] * n
        # scene_name: str
        return：
        """

        # for the final point cloud
        total_vertices = []
        total_verticesColor = []
        total_verticesNormal = []

        start = time.time()

        for view_pair in view_list:
            ref_view = view_pair["ref"]
            src_views = view_pair["src"]

            all_vis_infos = []
            geo_mask_sum = 0
            xyz_confidence_with_angle = 0

            t1 = time.time()

            # load the reference path
            ref_depth_est_path = join(self.depth_path, '{}_init.pfm'.format(ref_view))
            if self.save_temp and os.path.exists(join(self.tmp_path, '{}_init.pfm'.format(ref_view))):
                ref_depth_est_path = join(self.tmp_path, '{}_init.pfm'.format(ref_view))
                # ref_normal_est_path = join(self.tmp_path, '{}_normal.pfm'.format(ref_view))
            if not os.path.exists(ref_depth_est_path):
                warnings.warn("{} not exists".format(ref_depth_est_path))
                continue

            # load the reference camera parameters
            ref_intrinsics, ref_extrinsics, image_path = self.read_camera_parameters(
                join(self.depth_path, '{}.txt'.format(ref_view)), self.camera_scale)
            self.image_path = os.path.dirname(os.path.dirname(image_path))

            # load the reference image
            ref_img = self.read_img(join(self.image_path, '{}.jpg'.format(ref_view)), self.camera_scale)

            # load the estimated depth of the reference view
            ref_depth_est = read_pfm(ref_depth_est_path)[0]
            width, height = ref_depth_est.shape[1], ref_depth_est.shape[0]

            # load the estimated normal of the reference view
            normal_path = join(self.depth_path, '{}_normal.pfm'.format(ref_view))
            if os.path.exists(normal_path):
                ref_normal_est = self.read_normal(normal_path)
            else:
                print("create default normals for {}".format(ref_view))
                ref_normal_est = np.zeros([height, width, 3], dtype=np.float32)  # create default normals
                ref_normal_est[:, :, 2] = -1.0

            # load the photometric mask of the reference view
            confidence_path = join(self.depth_path, '{}_prob.pfm'.format(ref_view))
            if os.path.exists(confidence_path):
                confidence = read_pfm(confidence_path)[0]
            else:
                confidence = np.ones([height, width], dtype=np.float32)
            photo_mask = confidence > self.photometric_threshold

            # Ref 3d points
            x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
            x_ref_array, y_ref_array, depth_ref_array = x_ref.reshape([-1]), y_ref.reshape([-1]), ref_depth_est.reshape([-1])

            xyz_cam_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x_ref_array, y_ref_array, np.ones_like(x_ref_array))) * depth_ref_array)
            xyz_world_ref = np.matmul(np.linalg.inv(ref_extrinsics),
                                  np.vstack((xyz_cam_ref, np.ones_like(x_ref_array))))[:3]
            xyz_world_ref = xyz_world_ref.reshape([-1, height, width]).astype(np.float32)
            all_xyz_world = xyz_world_ref
            xyz_confidence_with_angle += np.ones_like(all_xyz_world)

            # Ref world normals
            ref_normal_est2 = ref_normal_est.transpose([2, 0, 1])
            ref_normal_est_world = np.matmul(np.linalg.inv(ref_extrinsics[:3, :3]), ref_normal_est2.reshape([3, -1]))
            ref_normal_est_world = ref_normal_est_world.reshape([3, height, width]).transpose([1, 2, 0])
            ref_normal_est_world = ref_normal_est_world/np.linalg.norm(ref_normal_est_world, axis=-1, keepdims=True)

            # vis information
            ref_idx = self.map_image_dict[ref_view]
            all_vis_infos.append((np.ones([height, width], dtype=np.int)) * ref_idx)
            geo_mask_sum += (np.ones([height, width], dtype=np.int32))

            for src_view in src_views[:self.fusion_num]:
                # load the source view
                src_depth_est_path = join(self.depth_path, '{}_init.pfm'.format(src_view))
                if self.save_temp and os.path.exists(join(self.tmp_path, '{}_init.pfm'.format(src_view))):
                    src_depth_est_path = join(self.tmp_path, '{}_init.pfm'.format(src_view))
                    # src_normal_est_path = join(self.tmp_path, '{}_normal.pfm'.format(src_view))
                if not os.path.exists(join(self.depth_path, '{}_init.pfm'.format(src_view))):
                    warnings.warn("{} not exists".format(
                        join(self.depth_path, '{}_init.pfm'.format(src_view))))
                    continue

                # camera parameters of the source view
                src_intrinsics, src_extrinsics, _ = self.read_camera_parameters(
                    join(self.depth_path, '{}.txt'.format(src_view)), self.camera_scale)
                # the estimated depth of the source view
                src_depth_est = read_pfm(src_depth_est_path)[0]
                # load the estimated normal of the source view
                src_normal_path = join(self.depth_path, '{}_normal.pfm'.format(src_view))
                if os.path.exists(src_normal_path):
                    src_normal_est = self.read_normal(src_normal_path)
                else:
                    src_normal_est = np.zeros([height, width, 3], dtype=np.float32)  # create default normals
                    src_normal_est[:, :, 2] = -1.0

                # geometric check
                geo_mask, depth_reprojected, src_depth_remove, xyz_world_src, angle_confidence_world_src = self.geometric_checker.check(
                    ref_depth_est, ref_normal_est, ref_intrinsics, ref_extrinsics, src_depth_est, src_normal_est,
                    src_intrinsics, src_extrinsics, confidence)

                if self.save_temp:
                    tmp_src_depth_path = join(self.tmp_path, '{}_init.pfm'.format(src_view))
                    tmp_dir = os.path.dirname(tmp_src_depth_path)
                    if not os.path.exists(tmp_dir):
                        os.mkdir(tmp_dir)
                    write_pfm(tmp_src_depth_path, src_depth_remove)
                    # write_pfm(tmp_src_normal_path, src_normal_remove)

                geo_mask_sum += geo_mask.astype(np.int32)
                all_xyz_world += (angle_confidence_world_src*xyz_world_src).astype(np.float32)
                xyz_confidence_with_angle += angle_confidence_world_src

                src_idx = self.map_image_dict[src_view]
                all_vis_infos.append(geo_mask.astype(np.int) * src_idx)

                del geo_mask, depth_reprojected, src_depth_remove, xyz_world_src

            depth_est_averaged = ref_depth_est
            avg_xyz_world = all_xyz_world / xyz_confidence_with_angle
            avg_xyz_world = avg_xyz_world.astype(np.float32)

            # at least N source views matched
            final_mask = np.array(geo_mask_sum >= self.min_geo_consist_num)

            if self.save_temp:
                tmp_ref_depth_path = join(self.tmp_path, '{}_init.pfm'.format(ref_view))
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                depth_est_averaged[~final_mask] = 0
                write_pfm(tmp_ref_depth_path, depth_est_averaged)
            t2 = time.time()

            if self.save_mask_or_not:
                self.save_mask(join(self.mask_path, "{}_final.png".format(ref_view)), final_mask)
            print("ref-view {}, fusion time:{}, photo/final-mask:{}/{}".format(ref_view, (t2 - t1), photo_mask.mean(), final_mask.mean()))

            # save the point cloud
            if final_mask.sum() < 10:
                print("ref-view {} no points left".format(ref_view))
                continue

            valid_points = final_mask
            all_vis_infos = np.array([vis_info[valid_points] for vis_info in all_vis_infos])
            valid_points_3c = np.repeat(np.expand_dims(valid_points, axis=0), 3, axis=0) # [3,H,W]
            avg_xyz_world = avg_xyz_world[valid_points_3c].reshape(3, -1)
            color = ref_img[valid_points]
            color = (color * 255).astype(int)
            valid_normal_world = ref_normal_est_world[valid_points]

            for i in range(0, avg_xyz_world.shape[1], self.skip_line):
                if len(all_vis_infos[0]) > 1:
                    xyz_arr = avg_xyz_world[:, i]
                    vis_arr = all_vis_infos[:, i]
                    if scene_range[0] < xyz_arr[0] < scene_range[1] and scene_range[2] < xyz_arr[1] < scene_range[3]:
                        visible_image_idxs = vis_arr[vis_arr > 0] - 1   
                        vertex = Vertex()
                        vertex.X = xyz_arr
                        vertex.views = visible_image_idxs 
                        vertex.viewSort()
                        vertex.confidence = [0.0] * len(visible_image_idxs)

                        total_vertices.append(vertex) 
                        total_verticesColor.append(color[i, :]) 
                        total_verticesNormal.append(valid_normal_world[i, :])  
                    else:
                        continue

        t3 = time.time()

        path_to_fused_mvs = join(self.output_path, '{}.mvs'.format(scene_name))
        path_to_fused_pc = join(self.output_path, '1/{}.ply'.format(scene_name))
        dir_ = os.path.dirname(path_to_fused_mvs)
        if not os.path.exists(dir_):
            os.mkdir(dir_)

        inter = Interface(self.sparse_path, self.image_path, path_to_fused_mvs, input_data=True)
        inter.Interface_Fused(total_vertices, total_verticesColor, total_verticesNormal, save_ply=True)
        t4 = time.time()

        print("\n==========> points save to {}".format(path_to_fused_mvs))
        print("==========> total fusion time:{}, save file time:{}\n".format((t3 - start), (t4 - t3)))

        shutil.rmtree(self.tmp_path)
        os.mkdir(self.tmp_path)

        return path_to_fused_pc


    def batch_fuse_depths(self):
        cnt = 0
        result_list = []

        for scene_block in self.scene_block_list:
            scene_range = scene_block["scene_range"]  # [min_x, max_x, min_y, max_y, min_z, max_z]
            view_list = scene_block["view_list"]  # a list ["ref", "src"] * n
            scene_name = 'scene_{}'.format(cnt)
            save_file_path = self.fuse_depths(scene_range, view_list, scene_name)
            result_list.append(save_file_path)

            border_path = join(self.output_path, scene_name + '.txt')
            save_border_as_file(border_path, scene_range)
            cnt = cnt + 1

        return result_list


if __name__ == '__main__':
    start = time.time()
    print("=================Fuse All Depths Start!=================")

    fdm = Fuse_Depth_Map(args.project_folder, args.sparse_path, args.mvs_path, args.output_path, args.fusion_num, args.min_geo_consist_num,args.photomatric_threshold,
                         args.position_threshold, args.depth_threshold,
                         args.camera_scale, args.cams_ori,
                         args.images_ori, args.implement, args.save_mask_or_not, args.pc_format)


    fdm.batch_fuse_depths()
    end = time.time()

    print("dense point clouds Saved.")
    print("------------ Cost {:.4f} min -------------".format((end - start) / 60.0))
    print("========== Fuse All Depths Finished! ==========")

