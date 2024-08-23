#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2022, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu

import argparse
import os
import numpy as np
import math
import cupy as cp
import torch
import torch.nn.functional as F



class ConsistencyChecker(object):
    def __init__(self, position_threshold, depth_threshold, normal_threshold, confidence_threshold, implement="cupy"):
        self.position_threshold = position_threshold
        self.depth_threshold = depth_threshold
        self.normal_threshold = math.cos(math.radians(normal_threshold))
        self.confidence_threshold = confidence_threshold
        print("normal_th:"+str(self.normal_threshold))

        self.implement = implement.lower()
        assert self.implement in ["numpy", "cupy", "torch"]


    def check_cupy(self, depth_ref, normal_ref, intrinsics_ref, extrinsics_ref,
              depth_src, normal_src, intrinsics_src, extrinsics_src, prob_map_ref):

        depth_ref = cp.array(depth_ref)
        normal_ref = cp.array(normal_ref)
        intrinsics_ref = cp.array(intrinsics_ref)
        extrinsics_ref = cp.array(extrinsics_ref)
        depth_src = cp.array(depth_src)
        normal_src = cp.array(normal_src)
        intrinsics_src = cp.array(intrinsics_src)
        extrinsics_src = cp.array(extrinsics_src)
        prob_map_ref = cp.array(prob_map_ref)
        valid_mask = depth_ref > 0

        # check the geometric consistency between the reference image and a source image
        width, height = depth_ref.shape[1], depth_ref.shape[0]
        x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
        x_ref = cp.array(x_ref)
        y_ref = cp.array(y_ref)

        # reprojection
        x_ref_array, y_ref_array = x_ref.reshape([-1]), y_ref.reshape([-1])

        # reference 3D space
        xyz_ref = cp.matmul(cp.linalg.inv(intrinsics_ref),
                            cp.vstack((x_ref_array, y_ref_array, cp.ones_like(x_ref_array)))
                            * depth_ref.reshape([-1]))
        # source 3D space
        # extrinsics_ref : Tcw
        xyz_src = cp.matmul(cp.matmul(extrinsics_src, cp.linalg.inv(extrinsics_ref)),
                            cp.vstack((xyz_ref, cp.ones_like(x_ref_array))))[:3]

        ptz = xyz_src[2].reshape([height, width]).astype(cp.float32)

        # source view x, y
        K_xyz_src = cp.matmul(intrinsics_src, xyz_src)
        xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

        # step2. reproject the source view points with source view depth estimation
        # find the depth estimation and normal of the source view

        x_src = (xy_src[0].reshape([height, width])+0.5).astype(cp.int)
        y_src = (xy_src[1].reshape([height, width])+0.5).astype(cp.int)
        sampled_depth_src = depth_src[y_src, x_src]
        sampled_normal_src = normal_src[y_src, x_src, :]

        # source 3D space
        # NOTE that we should use sampled source-view depth_here to project back
        xyz_src = cp.matmul(cp.linalg.inv(intrinsics_src),
                            cp.vstack((x_src.reshape([-1]), y_src.reshape([-1]), cp.ones_like(x_ref_array))) * sampled_depth_src.reshape([-1]))

        # world 3D space
        src_world = cp.matmul(cp.linalg.inv(extrinsics_src),
                                    cp.vstack((xyz_src, cp.ones_like(x_ref_array))))

        # reference 3D space
        xyz_reprojected = cp.matmul(extrinsics_ref, src_world)[:3]
        # source view x, y, depth
        depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(cp.float32)
        K_xyz_reprojected = cp.matmul(intrinsics_ref, xyz_reprojected)
        xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
        x_reprojected = xy_reprojected[0].reshape([height, width]).astype(cp.float32)
        y_reprojected = xy_reprojected[1].reshape([height, width]).astype(cp.float32)

        # check position
        dist = cp.sqrt((x_reprojected - x_ref) ** 2 + (y_reprojected - y_ref) ** 2)

        # check depth
        depth_diff = cp.abs(depth_reprojected - depth_ref)
        relative_depth_diff = depth_diff / depth_ref

        # check normal
        sampled_normal_src = sampled_normal_src.transpose([2, 0, 1]) # [H, W, 3] -> [3, H, W]
        normal_src_world = cp.matmul(cp.linalg.inv(extrinsics_src[:3, :3]), sampled_normal_src.reshape([3, -1]))
        normal_src_world = normal_src_world.reshape([3, height, width]).transpose([1, 2, 0])

        normal_ref = normal_ref.transpose([2, 0, 1])  # [H, W, 3] -> [3, H, W]
        normal_ref_world = cp.matmul(cp.linalg.inv(extrinsics_ref[:3, :3]), normal_ref.reshape([3, -1]))
        normal_ref_world = normal_ref_world.reshape([3, height, width]).transpose([1, 2, 0])
        # normal_ref_world = normal_ref_world / np.linalg.norm(normal_ref_world, axis=-1, keepdims=True)

        cos_sim = cp.sum(normal_ref_world*normal_src_world, axis=-1)
        cos_sim /= (cp.linalg.norm(normal_ref_world, axis=-1) * cp.linalg.norm(normal_src_world, axis=-1)) # [H, W]

        angle_confidence_world_src = np.repeat(np.expand_dims(cos_sim, axis=0), 3, axis=0) # [3,H,W]

        # consistency mask
        mask = cp.logical_and(dist < self.position_threshold, relative_depth_diff < self.depth_threshold)
        mask = cp.logical_and(mask, prob_map_ref > self.confidence_threshold)
        mask = cp.logical_and(mask, cos_sim > self.normal_threshold)
        mask = cp.logical_and(mask, valid_mask)

        depth_reprojected[~mask] = 0

        x_src_involved = (x_src[mask] + 0.5).astype(cp.int)
        y_src_involved = (y_src[mask] + 0.5).astype(cp.int)

        depth_src[y_src_involved, x_src_involved] = 0

        xyz_world_src = src_world[0:3].reshape([3, height, width]).astype(cp.float32)
        xyz_world_src[0][~mask] = 0
        xyz_world_src[1][~mask] = 0
        xyz_world_src[2][~mask] = 0

        angle_confidence_world_src[0][~mask] = 0
        angle_confidence_world_src[1][~mask] = 0
        angle_confidence_world_src[2][~mask] = 0
        angle_confidence_world_src[angle_confidence_world_src < 0] = 0

        return cp.asnumpy(mask), cp.asnumpy(depth_reprojected), cp.asnumpy(depth_src), cp.asnumpy(xyz_world_src), cp.asnumpy(angle_confidence_world_src)


    def check(self, depth_ref, normal_ref, intrinsics_ref, extrinsics_ref,
              depth_src, normal_src, intrinsics_src, extrinsics_src, prob_map_ref):
        
        ref_mask, ref_depth, src_depth, xyz_world_src, angle_confidence_world_src = self.check_cupy(depth_ref, normal_ref,
            intrinsics_ref, extrinsics_ref, depth_src, normal_src, intrinsics_src, extrinsics_src, prob_map_ref)
        
        return ref_mask, ref_depth, src_depth, xyz_world_src, angle_confidence_world_src

