# Copyright (c) 2022, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu

import os
import subprocess
import yaml
import argparse
import time
import shutil

# Indicate the openMVS binary directory
current_path = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(current_path, 'config.yaml')
OPENMVS_BIN = os.path.join(current_path, 'openmvs')


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class Create_Mesh:
    def __init__(self, recons_insert_distence=1.5, recons_decimate_ratio=1, refine_decimate_ratio=1, texture_decimate_ratio=1, refine_scale_times=1, config_info=None):
        # Indicate the openMVS binary directory
        # current_path = os.path.dirname(os.path.abspath(__file__))
        if config_info:
            self.config_path = config_info
        else:
            self.config_path = CONFIG_PATH
        self.OPENMVS_BIN = OPENMVS_BIN

        self.config = load_cfg(self.config_path)
        self.ReconstructMesh_config = self.config['RECONSTRUCTMESH']
        self.RefineMesh_config = self.config['REFINEMESH']
        self.TextureMesh_config = self.config['TEXTUREMESH']

        self.Recons_DistInsert = recons_insert_distence
        self.Recons_DecimateMesh = recons_decimate_ratio
        self.Refine_DecimateMesh = refine_decimate_ratio
        self.Texture_DecimateMesh = texture_decimate_ratio
        self.Refine_Scale_Times = refine_scale_times


    def create_folder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)


    def ReconstructMesh(self, workspace, strInputFileName, strOutputFileName):
        cmd = [os.path.join(self.OPENMVS_BIN, "ReconstructMesh"),
               "-w", workspace,
               "--export-type", self.ReconstructMesh_config['strExportType'],
               "--archive-type", str(self.ReconstructMesh_config['nArchiveType']),
               "--process-priority", str(self.ReconstructMesh_config['nProcessPriority']),
               "--max-threads", str(self.ReconstructMesh_config['nMaxThreads']),

               "-i", strInputFileName,
               "-o", strOutputFileName,
               "-d", str(self.Recons_DistInsert),
               "--constant-weight", str(self.ReconstructMesh_config['bUseConstantWeight']),
               "-f", str(self.ReconstructMesh_config['bUseFreeSpaceSupport']),
               "--thickness-factor", str(self.ReconstructMesh_config['fThicknessFactor']),
               "--quality-factor", str(self.ReconstructMesh_config['fQualityFactor']),

               "--decimate", str(self.Recons_DecimateMesh),
               "--target-face-num", str(self.ReconstructMesh_config['nTargetFaceNum']),
               "--remove-spurious", str(self.ReconstructMesh_config['fRemoveSpurious']),
               "--remove-spikes", str(self.ReconstructMesh_config['bRemoveSpikes']),
               "--close-holes", str(self.ReconstructMesh_config['nCloseHoles']),
               "--smooth", str(self.ReconstructMesh_config['nSmoothMesh']),

               "--mesh-file", self.ReconstructMesh_config['strMeshFileName'],
               "--mesh-export", str(self.ReconstructMesh_config['bMeshExport']),
               "--split-max-area", str(self.ReconstructMesh_config['fSplitMaxArea']),
               "--image-points-file", self.ReconstructMesh_config['strImagePointsFileName'],
               ]
        pIntrisics = subprocess.Popen(cmd)
        pIntrisics.wait()


    def RefineMesh(self, workspace, strInputFileName, strOutputFileName):

        cmd = [os.path.join(self.OPENMVS_BIN, "RefineMesh"),
               "-w", workspace,
               "--export-type", self.RefineMesh_config['strExportType'],
               "--archive-type", str(self.RefineMesh_config['nArchiveType']),
               "--process-priority", str(self.RefineMesh_config['nProcessPriority']),
               "--max-threads", str(self.RefineMesh_config['nMaxThreads']),

               "-i", strInputFileName,
               "-o", strOutputFileName,
               "--resolution-level", str(self.RefineMesh_config['nResolutionLevel']),
               "--min-resolution", str(self.RefineMesh_config['nMinResolution']),
               "--max-views", str(self.RefineMesh_config['nMaxViews']),
               "--decimate", str(self.Refine_DecimateMesh),
               "--close-holes", str(self.RefineMesh_config['nCloseHoles']),
               "--ensure-edge-size", str(self.RefineMesh_config['nEnsureEdgeSize']),
               "--max-face-area", str(self.RefineMesh_config['nMaxFaceArea']),
               "--scales", str(self.Refine_Scale_Times),
               "--scale-step", str(self.RefineMesh_config['fScaleStep']),
               "--reduce-memory", str(self.RefineMesh_config['nReduceMemory']),
               "--alternate-pair", str(self.RefineMesh_config['nAlternatePair']),
               "--regularity-weight", str(self.RefineMesh_config['fRegularityWeight']),
               "--rigidity-elasticity-ratio", str(self.RefineMesh_config['fRatioRigidityElasticity']),
               "--gradient-step", str(self.RefineMesh_config['fGradientStep']),
               "--planar-vertex-ratio", str(self.RefineMesh_config['fPlanarVertexRatio']),
               "--use-cuda", str(self.RefineMesh_config['bUseCUDA']),
               "--mesh-file", self.RefineMesh_config['strMeshFileName'],
               ]
        pIntrisics = subprocess.Popen(cmd)
        pIntrisics.wait()


    def TextureMesh(self, workspace, strInputFileName, strOutputFileName):

        cmd = [os.path.join(self.OPENMVS_BIN, "TextureMesh"),
               "-w", workspace,
               "--export-type", self.TextureMesh_config['strExportType'],
               "--archive-type", str(self.TextureMesh_config['nArchiveType']),
               "--process-priority", str(self.TextureMesh_config['nProcessPriority']),
               "--max-threads", str(self.TextureMesh_config['nMaxThreads']),

               "-i", strInputFileName,
               "-o", strOutputFileName,
               "--decimate", str(self.Texture_DecimateMesh),
               "--close-holes", str(self.TextureMesh_config['nCloseHoles']),
               "--resolution-level", str(self.TextureMesh_config['nResolutionLevel']),
               "--min-resolution", str(self.TextureMesh_config['nMinResolution']),
               "--outlier-threshold", str(self.TextureMesh_config['fOutlierThreshold']),
               "--cost-smoothness-ratio", str(self.TextureMesh_config['fRatioDataSmoothness']),
               "--global-seam-leveling", str(self.TextureMesh_config['bGlobalSeamLeveling']),
               "--local-seam-leveling", str(self.TextureMesh_config['bLocalSeamLeveling']),
               "--texture-size-multiple", str(self.TextureMesh_config['nTextureSizeMultiple']),
               "--patch-packing-heuristic", str(self.TextureMesh_config['nRectPackingHeuristic']),
               "--empty-color", str(self.TextureMesh_config['nColEmpty']),
               "--orthographic-image-resolution", str(self.TextureMesh_config['nOrthoMapResolution']),

               "--mesh-file", self.TextureMesh_config['strMeshFileName'],
               ]
        pIntrisics = subprocess.Popen(cmd)
        pIntrisics.wait()
    

    def run_mesh(self, input_path, output_path, file_name):
        """
        :param bbx_border: []
        :return:
        """
        workspace = output_path
        point_file = input_path + "/{}.mvs".format(file_name)
        mesh_file = output_path + "/scene_dense_mesh_{}.mvs".format(file_name)
        refine_mesh_file = output_path + "/scene_dense_mesh_refine_{}.mvs".format(file_name)
        texture_refine_mesh_file = output_path + "/scene_dense_mesh_refine_texture_{}.mvs".format(file_name)

        t1 = time.time()

        print("\n-----------> begin to reconstruct mesh")
        self.ReconstructMesh(workspace, point_file, mesh_file)
        t2 = time.time()

        print("\n-----------> begin to refine mesh")
        self.RefineMesh(workspace, mesh_file, refine_mesh_file)
        t3 = time.time()

        print("\n-----------> begin to texture mesh")
        self.TextureMesh(workspace, refine_mesh_file, texture_refine_mesh_file)
        t4 = time.time()

        print("---------------Cost {:.4f} min-------------".format((t4 - t1) / 60.0))

        return texture_refine_mesh_file.replace('.mvs', '.ply')


    def batch_run_mesh(self, input_path, output_path):
        blocks = []
        self.create_folder(output_path)

        list = os.listdir(input_path)
        for fname in list:
            if os.path.splitext(fname)[-1] == '.mvs':
                name = os.path.splitext(fname)[0]
                blocks.append(name)

        results_list = []
        if blocks is not None:
            print("====> total pc: {} blocks. ".format(len(blocks)))
            for block_name in blocks:
                texture_refine_mesh_file = self.run_mesh(input_path, output_path, block_name)
                results_list.append(texture_refine_mesh_file)
                results_list.append(texture_refine_mesh_file.replace('.ply', '.png'))
                print("====> mesh {} saved.".format(block_name))
        else:
            Exception("Cannot find file in {}. Please check it.".format(input_path))

        return results_list



PARSER = argparse.ArgumentParser()
PARSER.add_argument('--input_dir', default=r'F:\pipeline\pipeline_test\workspace_virtual_colmap_down\dense', help="the directory wich contains the pictures set.")
PARSER.add_argument('--output_dir', default=r'F:\pipeline\pipeline_test\workspace_virtual_colmap_down\dense\mesh', help="the directory wich will contain the resulting files.")
args = PARSER.parse_args()

if __name__ == '__main__':

    input_path = args.input_dir
    output_path = args.output_dir
    file_name = 'scene_dense_0'

    cm = Create_Mesh()
    cm.run_mesh(input_path, output_path, file_name)


