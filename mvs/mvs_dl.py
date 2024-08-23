#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Copyright (C) <2024> <Jin Liu and GPCV>

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

import os
import sys
import time


class MVS_Inference:
    def __init__(self, max_w, max_h, view_num=5, num_depth=384, min_interval=0.1, model_type='adamvs', pretrain_weight=None, display_depth=False):
        self.max_w = max_w
        self.max_h = max_h
        self.view_num = view_num
        self.num_depth = num_depth
        self.min_interval = min_interval
        self.model_type = model_type
        self.pretrain_weight = pretrain_weight
        self.display_depth = display_depth
        self.model_type = model_type.lower()

    def run(self, data_folder, mvs_path):
        if not os.path.exists(os.path.dirname(mvs_path)):
            os.mkdir(os.path.dirname(mvs_path))

        # casmvsnet
        if self.model_type in ["casmvsnet",  "ucsnet", "msrednet", "adamvs"]:
            predict_script = 'mvs/mvs_cas/predict.py'
            path = 'mvs/mvs_cas/checkpoints/{}/whu_omvs'.format(self.model_type)
        else:
            raise Exception("{}? Not implemented yet!".format(self.model_type))

        # default weight
        if self.pretrain_weight is None:
            list = os.listdir(path)
            for fname in list:
                if os.path.splitext(fname)[-1] == '.ckpt':
                    pretrain_weight = os.path.join(path, fname)
        else:
            pretrain_weight = self.pretrain_weight

        # run
        str_ = ('python {} --data_folder={} --output_folder={} --model={} --loadckpt={} --view_num={} --numdepth={} --max_w={} --max_h={} --min_interval={} --display={}'.format(
            predict_script, data_folder, mvs_path, self.model_type, pretrain_weight, self.view_num, self.num_depth,
            self.max_w, self.max_h, self.min_interval, self.display_depth))
        print(str_)
        os.system(str_)

