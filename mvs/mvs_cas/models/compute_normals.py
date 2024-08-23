import torch
import torch.nn as nn
import torch.nn.functional as F

Align_Corners_Range = False


class ComputeNormals(nn.Module):
    def __init__(self):
        super(ComputeNormals, self).__init__()

    def compute_3dpts_batch(self, pts, intrinsics):
        pts_shape = pts.shape 
        batchsize = pts_shape[0]
        height = pts_shape[1]
        width = pts_shape[2]

        y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=pts.device),
                                       torch.arange(0, width, dtype=torch.float32, device=pts.device)])
        y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
        y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

        xyz_ref = torch.matmul(torch.inverse(intrinsics),
                               torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(0) * pts.view(batchsize,
                                                                                                           -1).unsqueeze(1))

        xyz_ref = xyz_ref.view(batchsize, 3, height, width)
        xyz_ref = xyz_ref.permute(0, 2, 3, 1)

        return xyz_ref

    def compute_normal_by_depth(self, depth_est, ref_intrinsics, nei):
        depth_est_shape = depth_est.shape
        batchsize = depth_est_shape[0]
        height = depth_est_shape[1]
        width = depth_est_shape[2]

        pts_3d_map = self.compute_3dpts_batch(depth_est, ref_intrinsics)
        pts_3d_map = pts_3d_map.contiguous()

        pts_3d_map_ctr = pts_3d_map[:, nei:-nei, nei:-nei, :]
        pts_3d_map_x0 = pts_3d_map[:, nei:-nei, 0:-(2 * nei), :]
        pts_3d_map_y0 = pts_3d_map[:, 0:-(2 * nei), nei:-nei, :]
        pts_3d_map_x1 = pts_3d_map[:, nei:-nei, 2 * nei:, :]
        pts_3d_map_y1 = pts_3d_map[:, 2 * nei:, nei:-nei, :]
        pts_3d_map_x0y0 = pts_3d_map[:, 0:-(2 * nei), 0:-(2 * nei), :]
        pts_3d_map_x0y1 = pts_3d_map[:, 2 * nei:, 0:-(2 * nei), :]
        pts_3d_map_x1y0 = pts_3d_map[:, 0:-(2 * nei), 2 * nei:, :]
        pts_3d_map_x1y1 = pts_3d_map[:, 2 * nei:, 2 * nei:, :]

        diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
        diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
        diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
        diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
        diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
        diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
        diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
        diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

        pix_num = batchsize * (height - 2 * nei) * (width - 2 * nei)
        diff_x0 = diff_x0.view(pix_num, 3)
        diff_y0 = diff_y0.view(pix_num, 3)
        diff_x1 = diff_x1.view(pix_num, 3)
        diff_y1 = diff_y1.view(pix_num, 3)
        diff_x0y0 = diff_x0y0.view(pix_num, 3)
        diff_x0y1 = diff_x0y1.view(pix_num, 3)
        diff_x1y0 = diff_x1y0.view(pix_num, 3)
        diff_x1y1 = diff_x1y1.view(pix_num, 3)

        normals0 = F.normalize(
            torch.cross(diff_x1, diff_y1)) 
        normals1 = F.normalize(torch.cross(diff_x0, diff_y0)) 
        normals2 = F.normalize(torch.cross(diff_x0y1, diff_x0y0))
        normals3 = F.normalize(torch.cross(diff_x1y0, diff_x1y1))

        normal_vector = normals0 + normals1 + normals2 + normals3
        normal_vector = F.normalize(normal_vector)
        normal_map = normal_vector.view(batchsize, height - 2 * nei, width - 2 * nei, 3)

        normal_map = F.pad(normal_map, (0, 0, nei, nei, nei, nei), "constant", 0)

        return normal_map

    def compute_depth_by_normal(self, depth_map, normal_map, intrinsics, tgt_image, nei=1):
        depth_init = depth_map.clone()

        d2n_nei = 1 
        depth_map = depth_map[:, d2n_nei:-(d2n_nei), d2n_nei:-(d2n_nei)]
        normal_map = normal_map[:, d2n_nei:-(d2n_nei), d2n_nei:-(d2n_nei), :]

        depth_map_shape = depth_map.shape
        batchsize = depth_map_shape[0] 
        height = depth_map_shape[1]  
        width = depth_map_shape[2]  

        y_ctr, x_ctr = torch.meshgrid(
            [torch.arange(d2n_nei, height + d2n_nei, dtype=torch.float32, device=normal_map.device),
             torch.arange(d2n_nei, width + d2n_nei, dtype=torch.float32, device=normal_map.device)])
        y_ctr, x_ctr = y_ctr.contiguous(), x_ctr.contiguous()

        x_ctr_tile = x_ctr.unsqueeze(0).repeat(batchsize, 1, 1)
        y_ctr_tile = y_ctr.unsqueeze(0).repeat(batchsize, 1, 1)

        x0 = x_ctr_tile - d2n_nei
        y0 = y_ctr_tile - d2n_nei
        x1 = x_ctr_tile + d2n_nei
        y1 = y_ctr_tile + d2n_nei
        normal_x = normal_map[:, :, :, 0]
        normal_y = normal_map[:, :, :, 1]
        normal_z = normal_map[:, :, :, 2]

        fx, fy, cx, cy = intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]

        cx_tile = cx.unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)
        cy_tile = cy.unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)
        fx_tile = fx.unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)
        fy_tile = fy.unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)

        numerator = (x_ctr_tile - cx_tile) / fx_tile * normal_x + (y_ctr_tile - cy_tile) / fy_tile * normal_y + normal_z
        denominator_x0 = (x0 - cx_tile) / fx_tile * normal_x + (y_ctr_tile - cy_tile) / fy_tile * normal_y + normal_z
        denominator_y0 = (x_ctr_tile - cx_tile) / fx_tile * normal_x + (y0 - cy_tile) / fy_tile * normal_y + normal_z
        denominator_x1 = (x1 - cx_tile) / fx_tile * normal_x + (y_ctr_tile - cy_tile) / fy_tile * normal_y + normal_z
        denominator_y1 = (x_ctr_tile - cx_tile) / fx_tile * normal_x + (y1 - cy_tile) / fy_tile * normal_y + normal_z
        denominator_x0y0 = (x0 - cx_tile) / fx_tile * normal_x + (y0 - cy_tile) / fy_tile * normal_y + normal_z
        denominator_x0y1 = (x0 - cx_tile) / fx_tile * normal_x + (y1 - cy_tile) / fy_tile * normal_y + normal_z
        denominator_x1y0 = (x1 - cx_tile) / fx_tile * normal_x + (y0 - cy_tile) / fy_tile * normal_y + normal_z
        denominator_x1y1 = (x1 - cx_tile) / fx_tile * normal_x + (y1 - cy_tile) / fy_tile * normal_y + normal_z

        mask_x0 = denominator_x0 == 0
        denominator_x0 = denominator_x0 + 1e-3 * mask_x0.float()
        mask_y0 = denominator_y0 == 0
        denominator_y0 = denominator_y0 + 1e-3 * mask_y0.float()
        mask_x1 = denominator_x1 == 0
        denominator_x1 = denominator_x1 + 1e-3 * mask_x1.float()
        mask_y1 = denominator_y1 == 0
        denominator_y1 = denominator_y1 + 1e-3 * mask_y1.float()
        mask_x0y0 = denominator_x0y0 == 0
        denominator_x0y0 = denominator_x0y0 + 1e-3 * mask_x0y0.float()
        mask_x0y1 = denominator_x0y1 == 0
        denominator_x0y1 = denominator_x0y1 + 1e-3 * mask_x0y1.float()
        mask_x1y0 = denominator_x1y0 == 0
        denominator_x1y0 = denominator_x1y0 + 1e-3 * mask_x1y0.float()
        mask_x1y1 = denominator_x1y1 == 0
        denominator_x1y1 = denominator_x1y1 + 1e-3 * mask_x1y1.float()

        depth_map_x0 = numerator / denominator_x0 * depth_map
        depth_map_y0 = numerator / denominator_y0 * depth_map
        depth_map_x1 = numerator / denominator_y0 * depth_map
        depth_map_y1 = numerator / denominator_y0 * depth_map
        depth_map_x0y0 = numerator / denominator_x0y0 * depth_map
        depth_map_x0y1 = numerator / denominator_x0y1 * depth_map
        depth_map_x1y0 = numerator / denominator_x1y0 * depth_map
        depth_map_x1y1 = numerator / denominator_x1y1 * depth_map

        depth_x0 = depth_init
        depth_x0[:, d2n_nei:-(d2n_nei), :-(2 * d2n_nei)] = depth_map_x0
        depth_y0 = depth_init
        depth_y0[:, 0:-(2 * d2n_nei), d2n_nei:-(d2n_nei)] = depth_map_y0
        depth_x1 = depth_init
        depth_x1[:, d2n_nei:-(d2n_nei), 2 * d2n_nei:] = depth_map_x1
        depth_y1 = depth_init
        depth_y1[:, 2 * d2n_nei:, d2n_nei:-(d2n_nei)] = depth_map_y1
        depth_x0y0 = depth_init
        depth_x0y0[:, 0:-(2 * d2n_nei), 0:-(2 * d2n_nei)] = depth_map_x0y0
        depth_x1y0 = depth_init
        depth_x1y0[:, 0:-(2 * d2n_nei), 2 * d2n_nei:] = depth_map_x1y0
        depth_x0y1 = depth_init
        depth_x0y1[:, 2 * d2n_nei:, 0:-(2 * d2n_nei)] = depth_map_x0y1
        depth_x1y1 = depth_init
        depth_x1y1[:, 2 * d2n_nei:, 2 * d2n_nei:] = depth_map_x1y1

        tgt_image = tgt_image.permute(0, 2, 3, 1)
        tgt_image = tgt_image.contiguous()

        img_grad_x0 = tgt_image[:, d2n_nei:-d2n_nei, :-2 * d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei,
                                                                         d2n_nei:-d2n_nei, :]

        img_grad_x0 = F.pad(img_grad_x0, (0, 0, 0, 2 * d2n_nei, d2n_nei, d2n_nei), "constant", 1e-3)
        img_grad_y0 = tgt_image[:, :-2 * d2n_nei, d2n_nei:-d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei,
                                                                         d2n_nei:-d2n_nei, :]
        img_grad_y0 = F.pad(img_grad_y0, (0, 0, d2n_nei, d2n_nei, 0, 2 * d2n_nei), "constant", 1e-3)
        img_grad_x1 = tgt_image[:, d2n_nei:-d2n_nei, 2 * d2n_nei:, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei,
                                                                        :]
        img_grad_x1 = F.pad(img_grad_x1, (0, 0, 2 * d2n_nei, 0, d2n_nei, d2n_nei), "constant", 1e-3)
        img_grad_y1 = tgt_image[:, 2 * d2n_nei:, d2n_nei:-d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei,
                                                                        :]
        img_grad_y1 = F.pad(img_grad_y1, (0, 0, d2n_nei, d2n_nei, 2 * d2n_nei, 0), "constant", 1e-3)

        img_grad_x0y0 = tgt_image[:, :-2 * d2n_nei, :-2 * d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei,
                                                                        :]
        img_grad_x0y0 = F.pad(img_grad_x0y0, (0, 0, 0, 2 * d2n_nei, 0, 2 * d2n_nei), "constant", 1e-3)
        img_grad_x1y0 = tgt_image[:, :-2 * d2n_nei, 2 * d2n_nei:, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei,
                                                                       :]
        img_grad_x1y0 = F.pad(img_grad_x1y0, (0, 0, 2 * d2n_nei, 0, 0, 2 * d2n_nei), "constant", 1e-3)
        img_grad_x0y1 = tgt_image[:, 2 * d2n_nei:, :-2 * d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei,
                                                                       :]
        img_grad_x0y1 = F.pad(img_grad_x0y1, (0, 0, 0, 2 * d2n_nei, 2 * d2n_nei, 0), "constant", 1e-3)
        img_grad_x1y1 = tgt_image[:, 2 * d2n_nei:, 2 * d2n_nei:, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei,
                                                                      :]
        img_grad_x1y1 = F.pad(img_grad_x1y1, (0, 0, 2 * d2n_nei, 0, 2 * d2n_nei, 0), "constant", 1e-3)

        alpha = 0.1
        weights_x0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x0), 3))
        weights_y0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_y0), 3))
        weights_x1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x1), 3))
        weights_y1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_y1), 3))

        weights_x0y0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x0y0), 3))
        weights_x1y0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x1y0), 3))
        weights_x0y1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x0y1), 3))
        weights_x1y1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x1y1), 3))

        weights_sum = torch.sum(torch.stack(
            (weights_x0, weights_y0, weights_x1, weights_y1, weights_x0y0, weights_x1y0, weights_x0y1, weights_x1y1),
            0), 0)
        
        weights = torch.stack(
            (weights_x0, weights_y0, weights_x1, weights_y1, weights_x0y0, weights_x1y0, weights_x0y1, weights_x1y1),
            0) / weights_sum

        depth_map_avg = torch.sum(
            torch.stack((depth_x0, depth_y0, depth_x1, depth_y1, depth_x0y0, depth_x1y0, depth_x0y1, depth_x1y1),
                        0) * weights, 0)

        return depth_map_avg

    def forward(self, init_depth, img, intri_matrices):

        B, H, W = init_depth.shape[0], init_depth.shape[1], init_depth.shape[2],
        intri_matrices = torch.unbind(intri_matrices, 1)
        intrinsics = intri_matrices[0]
        resample_img = F.interpolate(img, [H, W], mode='bilinear', align_corners=Align_Corners_Range)

        rough_normal = self.compute_normal_by_depth(init_depth, intrinsics, nei=1)
        rough_normal = rough_normal.permute(0, 3, 1, 2) 

        return rough_normal

