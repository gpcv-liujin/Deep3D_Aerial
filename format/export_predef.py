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


def run_convert_predef(cam_txt, image_txt, out_cam_path, mode='rednet'):

    if not os.path.exists(out_cam_path):
        os.mkdir(out_cam_path)

    cams_dict = params_io.read_predef_cameras_text(cam_txt)
    photos_dict = params_io.read_predef_images_text(image_txt)

    if mode == 'rednet':
        # xrightyup Rwc twc
        for dict_key in photos_dict.keys():
            print(dict_key)
            dir = dict_key
            photo_params = photos_dict[dir]

            image_id = photo_params.image_id
            save_path = os.path.join(out_cam_path, str(image_id) + '.txt').replace("\\", "/") # todo
            print("export to %s.\n" % save_path)
            cam_params = cams_dict[photo_params.camera_id]
            params_io.write_params_for_rednet(save_path, cam_params, photo_params)

    elif mode == 'colmap':
        # XrightYdown Rcw tcw
        cameras_list = []
        photo_list = []

        for dict_key in cams_dict.keys():
            dir = dict_key
            cameras_params = cams_dict[dir]
            cameras_list.append(cameras_params)

        for dict_key in photos_dict.keys():
            dir = dict_key
            photo_params = photos_dict[dir]

            photo_id = photo_params.image_id
            img_name = photo_params.name
            r_mat = photo_params.rotation_matrix
            t_mat = photo_params.project_center

            cc = cameras.toCamera(r_mat, t_mat, camera_coordinate_type='XrightYup', rotation_type='Rwc', translation_tye='twc')
            cc2 = cameras.toCamera.to_camera_cw_xright_ydown(cc)

            qvec = cameras.rot2quat(cc2.rotation_matrix)
            tvec  = cc2.translation_vector
            cam_id = photo_params.camera_id

            photo_list.append([photo_id, qvec[3], qvec[0], qvec[1], qvec[2], tvec[0], tvec[1], tvec[2], cam_id, img_name])

        cams_txt = os.path.join(out_cam_path, 'cameras.txt').replace("\\", "/")
        image_txt = os.path.join(out_cam_path, 'images.txt').replace("\\", "/")
        points_txt = os.path.join(out_cam_path, 'points3D.txt').replace("\\", "/")
        params_io.write_params_for_colmap(cams_txt, image_txt, points_txt, cameras_list, photo_list)

    else:
        raise Exception("{}? Not implemented yet!".format(mode))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='rewrite camera files from predef.')

    parser.add_argument('--save_path', type=str, default=r'F:\pipeline\pipeline_test\workspace_1')
    parser.add_argument('--convert_format', type=str, default='colmap')

    args = parser.parse_args()

    # run
    run_convert_predef()







