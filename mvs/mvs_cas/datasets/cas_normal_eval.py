from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from datasets.data_io import *
from datasets.preprocess import *
from imageio import imread, imsave, imwrite


class MVSDataset(Dataset):
    def __init__(self, data_folder, mode, view_num, normalize, args, **kwargs):
        super(MVSDataset, self).__init__()
        self.data_folder = data_folder
        self.viewpair_path = self.data_folder + '/viewpair.txt'
        self.image_params_path = self.data_folder + '/images.txt'
        self.cam_params_path = self.data_folder + '/cameras.txt'
        self.image_path_path = self.data_folder + '/image_path.txt'

        self.mode = mode
        self.args = args
        self.view_num = view_num
        self.normalize = normalize
        self.min_interval = args.min_interval
        self.interval_scale = args.interval_scale
        self.num_depth = args.numdepth
        self.counter = 0
        assert self.mode in ["train", "val", "test"]

        self.cam_params_dict = read_cameras_text(self.cam_params_path)   # dict
        self.image_params_dict = read_images_text(self.image_params_path)  # dict
        self.image_paths, _ = read_images_path_text(self.image_path_path)   # dict
        self.sample_list = read_view_pair_text(self.viewpair_path, self.view_num)  # list [ref_view, src_views]
        self.sample_num = len(self.sample_list)


    def __len__(self):
        return len(self.sample_list)


    def read_img(self, filename):
        img = Image.open(filename)
        return img


    def read_depth(self, filename):
        # read pfm depth file
        depimg = imread(filename)
        depth_image = (np.float32(depimg) / 64.0)  # WHU MVS dataset
        # depth_image = np.array(read_pfm(filename)[0], dtype=np.float32)
        return np.array(depth_image)


    def create_cams(self, image_params, cam_params_dict, num_depth=384, min_interval=0.1):
        """
        read camera txt file  (XrightYup，[Rwc|twc])
        write camera for rednet  (XrightYdown, [Rcw|tcw]
        """

        # read camera txt file
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        extrinsics = np.zeros((4, 4), dtype=np.float32)

        # T
        O_xrightyup = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        R = np.matmul(image_params.rotation_matrix, O_xrightyup)  # Rwc, XrightYup to XrightYdown
        t = image_params.project_center      # twc
        extrinsics[0:3, 0:3] = R
        extrinsics[0:3, 3] = t
        extrinsics[3, 3] = 1.0
        extrinsics = np.linalg.inv(extrinsics)  # convert Twc to Tcw
        cam[0, :, :] = extrinsics

        # K
        cam_params = cam_params_dict[image_params.camera_id]
        fx = cam_params.focallength[0]
        fy = cam_params.focallength[1]
        x0 = cam_params.x0y0[0]
        y0 = cam_params.x0y0[1]
        cam[1][0][0] = fx
        cam[1][1][1] = fy
        cam[1][0][2] = x0
        cam[1][1][2] = y0
        cam[1][2][2] = 1

        cam[1][3][0] = image_params.depth[0]  # start
        # cam[1][3][1] = min_interval  # interval
        cam[1][3][1] = (image_params.depth[1]-image_params.depth[0])/num_depth  # interval
        cam[1][3][3] = image_params.depth[1] # end
        cam[1][3][2] = num_depth  # depth_sample_num

        return cam


    def __getitem__(self, idx):
        data = self.sample_list[idx]
        outimage = None
        outcam = None

        centered_images = []
        proj_matrices = []
        intri_matrices = []
        outlocation = []

        for view in range(self.view_num):
            # Images
            image_idx = data[view]
            image = self.read_img(self.image_paths[image_idx])
            image = np.array(image)

            # Cameras
            depth_interval = self.min_interval * self.interval_scale
            image_params = self.image_params_dict[image_idx]
            cam = self.create_cams(image_params, self.cam_params_dict, self.num_depth, depth_interval)

            # determine a proper scale to resize input
            scaled_image, scaled_cam = scale_input(image, cam, scale=self.args.resize_scale)
            # crop to fit network
            croped_image, croped_cam = crop_input(scaled_image, scaled_cam, max_h=self.args.max_h,
                                                  max_w=self.args.max_w, resize_scale=self.args.resize_scale)

            if view == 0:
                ref_img_path = self.image_paths[image_idx]
                outimage = croped_image
                outcam = croped_cam
                depth_min = croped_cam[1][3][0]
                depth_max = croped_cam[1][3][3]
                name = image_params.name
                vid = image_params.image_id
                h, w = croped_image.shape[0:2]
                outlocation.append(str(w))
                outlocation.append(str(h))
                outlocation.append(str(vid))
                outlocation.append(str(name))

            # scale cameras for building cost volume
            scaled_cam = scale_camera(croped_cam, scale=self.args.sample_scale)
            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics = scaled_cam[0, :, :]
            intrinsics = scaled_cam[1, 0:3, 0:3]
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            proj_matrices.append(proj_mat)
            intri_matrices.append(intrinsics)
            centered_images.append(center_image(croped_image, mode=self.normalize))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        intri_matrices = np.stack(intri_matrices)

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 4

        proj_matrices_ms = {
            "stage1": stage3_pjmats,
            "stage2": stage2_pjmats,
            "stage3": proj_matrices
        }

        stage2_intrimats = intri_matrices.copy()
        stage2_intrimats[:, :2, :] = intri_matrices[:, :2, :] / 2
        stage3_intrimats = intri_matrices.copy()
        stage3_intrimats[:, :2, :] = intri_matrices[:, :2, :] / 4

        intri_matrices_ms = {
            "stage1": stage3_intrimats,
            "stage2": stage2_intrimats,
            "stage3": intri_matrices
        }

        return {"imgs": centered_images,
                "proj_matrices": proj_matrices_ms,
                "intri_matrices": intri_matrices_ms,
                "depth_values": depth_values,
                "outimage": outimage,
                "outcam": outcam,
                "ref_image_path": ref_img_path,
                "outlocation": outlocation}




if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 128)
    item = dataset[50]

    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/val.txt', 'val', 3,
                         128)
    item = dataset[50]

    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/test.txt', 'test', 5,
                         128)
    item = dataset[50]

    # test homography here
    print(item.keys())
    print("imgs", item["imgs"].shape)
    print("depth", item["depth"].shape)
    print("depth_values", item["depth_values"].shape)
    print("mask", item["mask"].shape)

    ref_img = item["imgs"][0].transpose([1, 2, 0])[::4, ::4]
    src_imgs = [item["imgs"][i].transpose([1, 2, 0])[::4, ::4] for i in range(1, 5)]
    ref_proj_mat = item["proj_matrices"][0]
    src_proj_mats = [item["proj_matrices"][i] for i in range(1, 5)]
    mask = item["mask"]
    depth = item["depth"]

    height = ref_img.shape[0]
    width = ref_img.shape[1]
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    print("yy", yy.max(), yy.min())
    yy = yy.reshape([-1])
    xx = xx.reshape([-1])
    X = np.vstack((xx, yy, np.ones_like(xx)))
    D = depth.reshape([-1])
    print("X", "D", X.shape, D.shape)

    X = np.vstack((X * D, np.ones_like(xx)))
    X = np.matmul(np.linalg.inv(ref_proj_mat), X)
    X = np.matmul(src_proj_mats[0], X)
    X /= X[2]
    X = X[:2]

    yy = X[0].reshape([height, width]).astype(np.float32)
    xx = X[1].reshape([height, width]).astype(np.float32)
    import cv2

    warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
    warped[mask[:, :] < 0.5] = 0

    cv2.imwrite('../tmp0.png', ref_img[:, :, ::-1] * 255)
    cv2.imwrite('../tmp1.png', warped[:, :, ::-1] * 255)
    cv2.imwrite('../tmp2.png', src_imgs[0][:, :, ::-1] * 255)
