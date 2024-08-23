# Copyright (c) 2024, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import time
from datasets import find_dataset_def
from models import *
from utils import *
import sys
from datasets.data_io import load_pfm_utf8, save_pfm_utf8, write_red_cam
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth')
parser.add_argument('--model', default='adamvs', help='select model from [casmvs, ucs, msred, adamvs]')
parser.add_argument('--dataset', default='cas_normal_eval', help='select dataset') # casred_up_eval for casrednet_up; casred_refine_eval for msrednet_refine
parser.add_argument('--data_folder', default='/media/xxx/F1/pipeline/pipeline_test/workspace_munchen/export', help='test datapath')
parser.add_argument('--output_folder', default='/media/xxx/F1/pipeline/pipeline_test/workspace_munchen/dense/MVS', help='output dir')
parser.add_argument('--loadckpt', default='/media/xxx/X/liujin_densematching/multi_view_match/aerial_mvs_pipeline_draft_V1/mvs/checkpoints/casrs3net_vis2d_meitan_oblique_5_10/model_000019_0.1339_13.ckpt', help='load a specific checkpoint')

# input parameters
parser.add_argument('--view_num', type=int, default=5, help='Number of images (1 ref image and view_num - 1 view images).')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--max_w', type=int, default=3584, help='Maximum image width')
parser.add_argument('--max_h', type=int, default=4096, help='Maximum image height')
parser.add_argument('--min_interval', type=float, default=0.1, help='min_interval in the bottom stage')

parser.add_argument('--fext', type=str, default='.jpg', help='Type of images.')
parser.add_argument('--normalize', type=str, default='mean', help='methods of center_image, mean[mean var] or standard[0-1].') # attention: CasMVSNet [mean var];; CasREDNet [0-255]
parser.add_argument('--resize_scale', type=float, default=1.0, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=1, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--display', default=True, help='display depth images')

# Cascade parameters
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


# run MVS model to save depth maps and confidence maps
def predict_depth():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.data_folder, "val", args.view_num, args.normalize, args)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    # build model
    if args.model == 'casmvsnet':
        from models.cas_mvsnet import Infer_CascadeMVSNet
        model = Infer_CascadeMVSNet(num_depth=args.numdepth, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                              depth_intervals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                              share_cr=args.share_cr,
                              cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])

    elif args.model == 'ucsnet':
        from models.ucsnet import Infer_UCSNet
        model = Infer_UCSNet(lamb=1.5, num_depth=args.numdepth, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd])

    elif args.model == 'msrednet':
        from models.msrednet import Infer_CascadeREDNet
        model = Infer_CascadeREDNet(num_depth=args.numdepth, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                              depth_intervals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                              share_cr=args.share_cr,
                              cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])

    elif args.model == 'adamvs':
        from models.adamvs import Infer_AdaMVSNet
        model = Infer_AdaMVSNet(num_depth=args.numdepth, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                               depth_intervals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                               share_cr=args.share_cr,
                               cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])

    else:
        raise Exception("{}? Not implemented yet!".format(args.model))


    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    # checkpoint = torch.load(args.loadckpt, map_location='cpu')
    # state_dict = {}
    # for k, v in checkpoint['model'].items():
    #     state_dict[k.replace('module.', '')] = v
    #     print(k)
    # model.load_state_dict(state_dict, strict=True)

    model.eval()

    with torch.no_grad():
        # create output folder
        output_folder = args.output_folder
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        step = 0

        first_start_time = time.time()

        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            sample_cuda = tocuda(sample)

            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)
            depth_est = outputs["depth"]
            photometric_confidence = outputs["photometric_confidence"]
            duration = time.time()

            # save results
            depth_est = np.float32(np.squeeze(tensor2numpy(depth_est)))
            prob = np.float32(np.squeeze(tensor2numpy(photometric_confidence)))
            ref_cam = np.squeeze(tensor2numpy(sample["outcam"]))
            ref_path = np.squeeze(sample["ref_image_path"])
            out_location = np.squeeze(sample["outlocation"])
            print(out_location)
            name = os.path.splitext(out_location[3])[0]

            # paths
            init_depth_map_path = output_folder + ('/%s_init.pfm' % name)
            prob_map_path = output_folder + ('/%s_prob.pfm' % name)
            out_ref_image_path = output_folder + ('/%s.jpg' % name)
            out_ref_cam_path = output_folder + ('/%s.txt' % name)

            dir = os.path.dirname(init_depth_map_path)
            if not os.path.exists(dir):
                os.mkdir(dir)

            if args.display:
                # color output
                size1 = len(depth_est)
                size2 = len(depth_est[1])
                e = np.ones((size1, size2), dtype=np.float32)
                out_init_depth_image = e * 36000 - depth_est
                if not os.path.exists(output_folder + '/color/'):
                    os.mkdir(output_folder + '/color/')
                color_depth_map_path = output_folder + ('/color/%s_init.png' % name)
                color_prob_map_path = output_folder + ('/color/%s_prob.png' % name)
                dir = os.path.dirname(color_depth_map_path)
                if not os.path.exists(dir):
                    os.mkdir(dir)

                for i in range(out_init_depth_image.shape[1]):
                    col = out_init_depth_image[:, i]
                    col[np.isinf(col)] = np.nan
                    col[np.isnan(col)] = np.nanmin(col) - 1
                    out_init_depth_image[:, i] = col

                plt.imsave(color_depth_map_path, out_init_depth_image, format='png')
                plt.imsave(color_prob_map_path,  np.nan_to_num(prob).clip(0, 1), format='png')

            # save output
            save_pfm_utf8(init_depth_map_path, depth_est)
            save_pfm_utf8(prob_map_path, prob)
            # plt.imsave(out_ref_image_path, ref_image, format='png')
            # cv2.imwrite(out_ref_image_path, ref_image)
            write_red_cam(out_ref_cam_path, ref_cam, out_location, ref_path)
            del outputs, sample_cuda

            step = step + 1
            save_tesult_time = time.time()
            print('depth inference {} finished, image {} finished, ({:3f}s and {:3f} sec/step)'.format(step, name, duration-start_time, save_tesult_time-duration))

        print("final, total_cnt = {}, total_time = {:3f}".format(step, time.time() - first_start_time))


if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    predict_depth()

