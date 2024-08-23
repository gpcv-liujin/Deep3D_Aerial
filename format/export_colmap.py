#!/usr/bin/env python
# Copyright (c) 2024, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu

import sys
sys.path.append("..")
import math
import numpy as np
import os
import argparse
from format import cameras
from IO import params_io
from pycolmap.read_write_model import read_model, qvec2rotmat


def run_convert_colmap(sparse_path, output_path, mode='predef'):

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Read and write model
    cams, images, point3ds = read_model(sparse_path)

    cameras_list = []
    photo_list = []

    camera_info = dict()
    for cam in cams.values():
        pinhole_params = cam.params
        camera_info[cam.id] = np.array([[pinhole_params[0], 0, pinhole_params[2]],
                                        [0, pinhole_params[1], pinhole_params[3]],
                                        [0, 0, 1]])
        cameras_param = cameras.Camera(camera_id=cam.id, camera_model=cam.model, size=[cam.width, cam.height], pixelsize=0,
                                       focallength=[cam.params[0], cam.params[1]], x0y0=[cam.params[2], cam.params[3]],
                                       distortion=cam.params[4:] if len(cam.params[4:])>4 else [0, 0, 0, 0])  # distortion=cam.params[4:]

        cameras_list.append(cameras_param)

    for img in images.values():
        intrinsic = camera_info[img.camera_id]
        rotation_matrix = qvec2rotmat(img.qvec).reshape(3, 3)
        translate_vector = img.tvec.reshape(-1, 1)
        extrinsic = np.concatenate([rotation_matrix, translate_vector], axis=1)

        obj_pt = np.array([[point3ds[idx].xyz[0], point3ds[idx].xyz[1], point3ds[idx].xyz[2], 1]
                           for idx in img.point3D_ids if idx != -1])

        if obj_pt.shape[0] > 0:
            homo_extrinsic = np.eye(4)
            homo_extrinsic[:3, :4] = extrinsic

            homo_intrinsic = np.eye(4)
            homo_intrinsic[:3, :3] = intrinsic

            projection_matrix = np.matmul(homo_intrinsic, homo_extrinsic)
            calculate_img_pt = np.matmul(projection_matrix, obj_pt.T).T
            depth = calculate_img_pt[:, 2]

            # depth_range = [np.min(depth), np.max(depth)]
            min_depth, max_depth = np.percentile(depth, [0.1, 99.9])
            rel_depth = (max_depth - min_depth) / 64.0
            depth_range = [min_depth - rel_depth, max_depth + rel_depth]

            # camera ori
            cc = cameras.toCamera(rotation_matrix, translate_vector, camera_coordinate_type='XrightYdown',
                                  rotation_type='Rcw',
                                  translation_tye='tcw')
            cc2 = cameras.toCamera.to_camera_wc_xright_yup(cc)

            photo_param = cameras.Photo(image_id=img.id, camera_id=img.camera_id, rotation_matrix=cc2.rotation_matrix,
                                        project_center=cc2.translation_vector, depth=depth_range, name=img.name)
            photo_list.append(photo_param)

    if mode == 'predef':
        cams_txt = os.path.join(output_path, 'cameras.txt').replace("\\", "/")
        image_txt = os.path.join(output_path, 'images.txt').replace("\\", "/")
        params_io.write_params_for_predef(cams_txt, image_txt, cameras_list, photo_list)

    elif mode == 'rednet':

        camera_info = dict()
        for cam in cameras_list:
            camera_info[cam.camera_id] = cam
        for photo in photo_list:
            cam_params = camera_info[photo.camera_id]
            out_cam_path = os.path.join(output_path, photo.name[:len(photo.name)-4] + '.txt').replace("\\", "/")
            params_io.write_params_for_rednet(out_cam_path, cam_params, photo)

    else:
        raise Exception("{}? Not implemented yet!".format(mode))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='rewrite camera files from colmap.')

    parser.add_argument('--sparse_path', type=str, default=r'F:\pipeline\pipeline_test\workspace_1\sparse')
    parser.add_argument('--output_path', type=str, default=r'F:\pipeline\pipeline_test\workspace_1\pair')
    parser.add_argument('--convert_format', type=str, default='predef')

    args = parser.parse_args()

    run_convert_colmap(args.sparse_path, args.output_path, mode=args.convert_format)





