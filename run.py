#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Copyright (C) 2024 <Jin Liu and GPCV>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author: Jin Liu
"""


import time
import argparse

from format import export_predef, export_colmap
from IO.params_io import *
from tools.utils import *
from mvs.mvs_dl import MVS_Inference
from fuse.fusion_3d_normal import Fuse_Depth_Map
from mesh.createmesh import Create_Mesh
from dsm.mesh_source import mesh2dsm
from dsm.pc_source import pc2dsm
from tools.log import Save_Logger
from pycolmap.view_selection import calculate_block_border, select_view_based_viewed_points


class AerialMVS:
    def __init__(self, project_folder, export_folder, config_folder):

        # project folder
        self.project_folder = project_folder
        self.export_path = export_folder
        self.log_path = join(self.project_folder, "logs")

        # dense folder
        self.dense_path = join(self.project_folder, "dense")
        self.mvs_path = join(self.dense_path, "MVS")
        self.fusion_path = join(self.dense_path, "fusion")
        self.mesh_path = join(self.project_folder, "mesh")
        self.dsm_path = join(self.project_folder, "dsm")

        # production folder
        self.production_path = join(self.project_folder, "production")
        self.production_pc_path = join(self.production_path, "Point_Cloud")
        self.production_mesh_path = join(self.production_path, "Mesh")
        self.production_dsm_path = join(self.production_path, "DSM")
        self.production_resample_mesh_path = join(self.production_path, "Resample_Mesh")
        self.production_resample_pc_path = join(self.production_path, "Resample_PC")

        # config
        config = parseArguments(config_folder, format='yaml')
        print(config)
        Save_Logger(self.log_path)


        # preprocess config
        self.preprocess_config = config['PREPROCESS']
        # # data type
        self.fext = self.preprocess_config["fext"]                        # data format
        self.cams_ori = self.preprocess_config["cams_ori"]                # Camera orientation [0 XrightYdown; 1 XrightYup]
        self.rotation_ori = self.preprocess_config["rotation_ori"]        # rotation orientation
        self.translation_ori = self.preprocess_config["translation_ori"]  # translation orientation
        # # image size
        self.image_w = self.preprocess_config["image_w"]
        self.image_h = self.preprocess_config["image_h"]
        self.image_scale = self.preprocess_config["image_scale"]

        # view_selection config
        self.viewselection_config = config['VIEWSELECTION']
        self.run_view_selection = self.viewselection_config["run_view_selection"]  
        self.view_selection_mode = self.viewselection_config["view_selection_mode"]
        self.scene_block_size = self.viewselection_config["scene_block_size"]
        self.bbx_border_scene = self.viewselection_config["bbx_border_scene"]
        self.block_overlap = self.viewselection_config["block_overlap"]

        # dense matching config
        self.densematch_config = config['DENSEMATCH']
        self.run_mvs = self.densematch_config["run_mvs"]                   
        self.view_num = self.densematch_config["view_num"]
        self.num_depth = self.densematch_config["num_depth"]
        self.max_w = int(self.image_w * self.image_scale)
        self.max_h = int(self.image_h * self.image_scale)
        self.min_interval = self.densematch_config["min_interval"]
        self.model_type = self.densematch_config["model_type"]
        self.pretrain_weight = self.densematch_config["pretrain_weight"]
        self.display_depth = self.densematch_config["display_depth"]

        # fusion config
        self.fusion_config = config['FUSION']
        self.run_fusion_depth = self.fusion_config["run_depth_fusion"]    
        self.fusion_num = self.fusion_config["fusion_num"]
        self.confidence_ratio = self.fusion_config["photomatric_threshold"]
        self.geo_consist_num = self.fusion_config["geo_consist_num"]
        self.position_threshold = self.fusion_config["position_threshold"]
        self.depth_threshold = self.fusion_config["depth_threshold"]
        self.normal_threshold = self.fusion_config["normal_threshold"]
        self.pc_format = self.fusion_config["pc_format"]

        # mesh config
        self.mesh_config = config['CREATEMESH']
        self.run_create_mesh = self.mesh_config["run_create_mesh"] 
        self.recons_insert_distance = self.mesh_config["recons_insert_distance"]
        self.recons_decimate_ratio = self.mesh_config["recons_decimate_ratio"]
        self.refine_decimate_ratio = self.mesh_config["refine_decimate_ratio"]
        self.texture_decimate_ratio = self.mesh_config["texture_decimate_ratio"]
        self.refine_scale_times = self.mesh_config["refine_scale_times"]

        # dsm config
        self.dsm_config = config['CREATEDSM']
        self.run_create_dsm = self.dsm_config["run_create_dsm"]
        self.dsm_source = self.dsm_config["dsm_source"]
        self.pc_select_method = self.dsm_config["pc_select_method"]
        self.pc_interpolation_method = self.dsm_config["pc_interpolation_method"]
        self.dsm_uint = self.dsm_config["dsm_uint"]
        self.dsm_size = self.dsm_config["dsm_size"]
        self.bbx_border_dsm = self.dsm_config["bbx_border_dsm"]


    def select_view(self, database_path, sparse_path, export_path, run_view_selection=True):
        if run_view_selection:
            start = time.time()
            print("************ View Selection Start! ************")

            # export predef
            export_colmap.run_convert_colmap(sparse_path, export_path, mode='predef')

            # export block information
            range_list, scene_border = calculate_block_border(sparse_path, base_size=self.scene_block_size, base_overlap=self.block_overlap, bbx_border=self.bbx_border_scene)
            # select view
            total_ref_view_in_range, total_viewpair = select_view_based_viewed_points(database_path, sparse_path, range_list, mode=self.view_selection_mode)

            # export view pair files
            pair_txt = join(export_path, "viewpair.txt")
            block_txt = join(export_path, "blocks.txt")
            scene_border_txt = join(export_path, "scene_border.txt")

            write_pair_text(pair_txt, total_viewpair)
            write_block_text(block_txt, total_ref_view_in_range)
            save_border_as_file(scene_border_txt, scene_border)

            end = time.time()
            print("------------ Cost {:.4f} min -------------".format((end - start) / 60.0))
            print("************ View Selection Finished! ************")
        else:
            print("************ View Selection Skip! ************")


    def dense_match(self, data_folder, mvs_path, run_mvs=True):
        if run_mvs:
            start = time.time()
            print("************ MVS Start! ************")
            mi = MVS_Inference(self.max_w, self.max_h, self.view_num, self.num_depth, self.min_interval, self.model_type, self.pretrain_weight, self.display_depth)
            mi.run(data_folder, mvs_path)
            end = time.time()
            print("---------------Cost {:.4f} min-------------".format((end - start) / 60.0))
            print("************ MVS Finished! ************")
        else:
            print("************ MVS Skip! ************")

    def fuse_depth_map(self, view_pair_path, sparse_path, mvs_path, fusion_path, run_fusion_depth=True):
        if run_fusion_depth:
            start = time.time()
            print("************ Fuse All Depths Start! ************")
            fdm = Fuse_Depth_Map(view_pair_path, sparse_path, mvs_path, fusion_path, self.fusion_num, self.geo_consist_num, self.confidence_ratio, pc_format=self.pc_format)
            pc_results_list = fdm.batch_fuse_depths()
            print("dense point clouds Saved.")
            end = time.time()
            print("------------ Cost {:.4f} min -------------".format((end - start) / 60.0))
            print("************ Fuse All Depths Finished! ************")
            self.move_production(pc_results_list, self.production_pc_path)
        else:
            print("************ Fuse Depths Skip! ************")


    def create_mesh(self, input_path, output_path, run_create_mesh=True):
        if run_create_mesh:
            start = time.time()
            print("************  Create Mesh Start! ************ ")
            cm = Create_Mesh(self.recons_insert_distance, self.recons_decimate_ratio, self.refine_decimate_ratio,
                             self.texture_decimate_ratio, self.refine_scale_times)
            mesh_results_list = cm.batch_run_mesh(input_path, output_path)

            # write to file
            with open(output_path + "/result_list.txt", "w") as f:
                for b in mesh_results_list:
                    f.write(os.path.join(output_path + b))
                    f.write('\n')

            self.move_production(mesh_results_list, self.production_mesh_path)
            end = time.time()
            print("---------------Cost {:.4f} min-------------".format((end - start) / 60.0))
            print("************  Create Mesh Finished! ************")
        else:
            print("************ Create Mesh Skip! ************")


    def create_dsm(self, dsm_source, dsm_result_path, bbx_border_dsm, run_create_dsm=True):
        if run_create_dsm:
            start = time.time()
            print("************ Create DSM Start! ************")
            # define border
            if bbx_border_dsm is None:
                bbx_border_dsm = load_border_from_file(join(self.export_path, 'scene_border.txt'))  # [Xmin, Xmax, Ymin, Ymax, Zmin, Zmax]

            # define GSD
            dsm_uint = self.dsm_uint if self.dsm_uint is not None else [0.1, 0.1]

            # define size
            dsm_size = self.dsm_size if self.dsm_size is not None else \
                [math.ceil((bbx_border_dsm[1] - bbx_border_dsm[0]) / dsm_uint[0]),
                 math.ceil((bbx_border_dsm[3] - bbx_border_dsm[2]) / dsm_uint[1])]

            # choose source
            if dsm_source == "mesh":
                input_path = self.production_mesh_path
                output_path = dsm_result_path + '_from_Mesh'
                mkdir_if_not_exist(output_path)

                dm = mesh2dsm.DSM_from_Mesh(input_path, output_path, dsm_uint, dsm_size)
                dm.create(bbx_border_dsm)

            elif dsm_source == "pc":
                input_path = self.production_pc_path
                output_path = dsm_result_path + '_from_PC'
                mkdir_if_not_exist(output_path)

                cdd = pc2dsm.DSM_from_PC(input_path, output_path, dsm_uint, dsm_size, self.pc_select_method, self.pc_interpolation_method)
                cdd.create(bbx_border_dsm)

            else:
                raise Exception("dsm source: {}? Not implemented yet!".format(dsm_source))

            end = time.time()
            print("---------------Cost {:.4f} min-------------".format((end - start) / 60.0))
            print("************ Create DSM Finished! ************")
        else:
            print("************ Create DSM Skip! ************")


    def move_production(self, result_list, save_path):
        mkdir_if_not_exist(os.path.dirname(save_path))
        mkdir_if_not_exist(save_path)
        if result_list is not None:
            for result in result_list:
                result_name = os.path.basename(result)
                new_path = join(save_path, result_name)
                if os.path.exists(result):
                    shutil.copy(result, new_path)


    def run_dense(self):
        print("======================= Aerial MVS Pipeline =======================")
        total_start = time.time()

        # calculate matching pairs for mvs
        self.select_view(join(self.export_path, 'database.db'), join(self.export_path, 'sparse_model'), self.export_path, run_view_selection=self.run_view_selection)

        # MVS inference
        self.dense_match(self.export_path, self.mvs_path, run_mvs=self.run_mvs)

        # depth map fusion
        self.fuse_depth_map(self.export_path, join(self.export_path, 'sparse_model'), self.mvs_path, self.fusion_path, run_fusion_depth=self.run_fusion_depth)

        # production
        self.create_mesh(self.fusion_path, self.mesh_path, run_create_mesh=self.run_create_mesh)

        self.create_dsm(self.dsm_source, self.production_dsm_path, self.bbx_border_dsm, run_create_dsm=self.run_create_dsm)

        total_end = time.time()
        print(
            "===================== Totally Cost {} min ======================".format((total_end - total_start) / 60.0))



parser = argparse.ArgumentParser(description='Aerial MVS Pipeline')
parser.add_argument('--workspace_folder', type=str, default=r'X:\1_push2git\github_release\Deep3D_Aerial\example\workspace')
parser.add_argument('--data_folder', type=str, default=r'X:\1_push2git\github_release\Deep3D_Aerial\example\workspace\export')
parser.add_argument('--config', type=str, default=r'X:\1_push2git\github_release\Deep3D_Aerial\example\workspace\config.yaml')
parser.add_argument('--cuda', type=str, default='0')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


if __name__ == '__main__':

    workspace_folder = args.workspace_folder
    data_folder = args.data_folder
    config_folder = args.config

    # init
    amvs = AerialMVS(workspace_folder, data_folder, config_folder)
    # run dense reconstruction
    amvs.run_dense()

