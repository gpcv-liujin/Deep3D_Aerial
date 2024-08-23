import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss



def uncertainty_aware_samples(cur_depth, exp_var, ndepth, device, dtype, shape):
    eps = 1e-12
    if cur_depth.dim() == 2:
        #must be the first stage
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )
        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) # (B, D)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) # (B, D, H, W)
    else:
        low_bound = cur_depth - exp_var
        high_bound = cur_depth + exp_var

        assert ndepth > 1

        step = (high_bound - low_bound) / (float(ndepth) - 1)
        new_samps = []
        for i in range(int(ndepth)):
            new_samps.append(low_bound + step * i + eps)

        depth_range_samples = torch.cat(new_samps, 1)

    return depth_range_samples


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels=8):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


def compute_depth(feats, proj_mats, depth_samps, cost_reg, lamb, is_training=False):
    '''

    :param feats: [(B, C, H, W), ] * num_views
    :param proj_mats: [()]
    :param depth_samps:
    :param cost_reg:
    :param lamb:
    :return:
    '''

    proj_mats = torch.unbind(proj_mats, 1)
    num_views = len(feats)
    num_depth = depth_samps.shape[1]

    assert len(proj_mats) == num_views, "Different number of images and projection matrices"

    ref_feat, src_feats = feats[0], feats[1:]
    ref_proj, src_projs = proj_mats[0], proj_mats[1:]

    ref_volume = ref_feat.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    volume_sum = ref_volume
    volume_sq_sum = ref_volume ** 2
    del ref_volume

    for src_fea, src_proj in zip(src_feats, src_projs):
        warped_volume = homo_warping_float(src_fea, src_proj, ref_proj, depth_samps)

        if is_training:
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        else:
            volume_sum += warped_volume
            volume_sq_sum += warped_volume.pow_(2)
        del warped_volume
    volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

    prob_volume_pre = cost_reg(volume_variance).squeeze(1)
    prob_volume = F.softmax(prob_volume_pre, dim=1)
    depth = depth_regression(prob_volume, depth_values=depth_samps)

    with torch.no_grad():
        prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                            stride=1, padding=0).squeeze(1)
        depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                              dtype=torch.float)).long()
        depth_index = depth_index.clamp(min=0, max=num_depth - 1)
        prob_conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

    samp_variance = (depth_samps - depth.unsqueeze(1)) ** 2
    exp_variance = lamb * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5

    return {"depth": depth, "photometric_confidence": prob_conf, 'variance': exp_variance}


class UCSNet(nn.Module):
    def __init__(self, lamb=1.5, ndepths=[64, 32, 8], grad_method="detach", arch_mode="unet", base_chs=[8, 8, 8]):
        super(UCSNet, self).__init__()
        self.ndepths = ndepths
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.base_chs = base_chs
        self.lamb = lamb
        self.num_stage = len(ndepths)

        self.stage_infos = {"stage1": 4.0,
                         "stage2": 2.0,
                         "stage3": 1.0 }

        self.feature_extraction = FeatureNet_mvsnet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)

        self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature_extraction.out_channels[i],
                                                             base_channels=self.base_chs[i]) for i in range(self.num_stage)])


    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -2].cpu().numpy())
        depth_interval = float(depth_values[0, -1].cpu().numpy())
        depth_range = depth_values[:, 0:-1]

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.shape[1]):
            img = imgs[:, nview_idx]
            features.append(self.feature_extraction(img))

        outputs = {}
        depth, cur_depth, exp_var = None, None, None


        for stage_idx in range(self.num_stage):

            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]

            cur_h = img.shape[2] // int(stage_scale)
            cur_w = img.shape[3] // int(stage_scale)

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                    exp_var = exp_var.detach()
                else:
                    cur_depth = depth

                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [cur_h, cur_w], mode='bilinear')
                exp_var = F.interpolate(exp_var.unsqueeze(1), [cur_h, cur_w], mode='bilinear')

            else:
                cur_depth = depth_range

            depth_range_samples = uncertainty_aware_samples(cur_depth=cur_depth,
                                                            exp_var=exp_var,
                                                            ndepth=self.ndepths[stage_idx],
                                                            dtype=img[0].dtype,
                                                            device=img[0].device,
                                                            shape=[img.shape[0], cur_h, cur_w])

            outputs_stage = compute_depth(features_stage, proj_matrices_stage,
                                          depth_samps=depth_range_samples,
                                          cost_reg=self.cost_regularization[stage_idx],
                                          lamb=self.lamb,
                                          is_training=self.training)

            depth = outputs_stage['depth']
            exp_var = outputs_stage['variance']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


class Infer_UCSNet(nn.Module):
    def __init__(self, lamb=1.5, ndepths=[64, 32, 8], grad_method="detach", arch_mode="unet", base_chs=[8, 8, 8]):
        super(Infer_UCSNet, self).__init__()
        self.ndepths = ndepths
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.base_chs = base_chs
        self.lamb = lamb
        self.num_stage = len(ndepths)

        self.stage_infos = {"stage1": 4.0,
                         "stage2": 2.0,
                         "stage3": 1.0 }

        self.feature_extraction = FeatureNet_mvsnet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)

        self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature_extraction.out_channels[i],
                                                             base_channels=self.base_chs[i]) for i in range(self.num_stage)])


    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / self.num_depth
        depth_range = depth_values

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.shape[1]): 
            img = imgs[:, nview_idx]
            features.append(self.feature_extraction(img))

        outputs = {}
        depth, cur_depth, exp_var = None, None, None


        for stage_idx in range(self.num_stage):

            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]

            cur_h = img.shape[2] // int(stage_scale)
            cur_w = img.shape[3] // int(stage_scale)

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                    exp_var = exp_var.detach()
                else:
                    cur_depth = depth

                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [cur_h, cur_w], mode='bilinear')
                exp_var = F.interpolate(exp_var.unsqueeze(1), [cur_h, cur_w], mode='bilinear')

            else:
                cur_depth = depth_range

            depth_range_samples = uncertainty_aware_samples(cur_depth=cur_depth,
                                                            exp_var=exp_var,
                                                            ndepth=self.ndepths[stage_idx],
                                                            dtype=img[0].dtype,
                                                            device=img[0].device,
                                                            shape=[img.shape[0], cur_h, cur_w])

            outputs_stage = compute_depth(features_stage, proj_matrices_stage,
                                          depth_samps=depth_range_samples,
                                          cost_reg=self.cost_regularization[stage_idx],
                                          lamb=self.lamb,
                                          is_training=self.training)

            depth = outputs_stage['depth']
            exp_var = outputs_stage['variance']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs

