#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2024, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu


import os
import shutil
import json
import yaml


def parseArguments_json(config_path):
    with open(config_path, encoding='utf-8') as json_file:
        tp = json.load(json_file)
    return tp


def parseArguments(config_path, format='yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        if format == 'yaml':
            tp = yaml.load(f, Loader=yaml.FullLoader)
        elif format == 'json':
            tp = json.load(f)
        else:
            tp = []
        return tp


def get_current_paths(folder, fext):
    # Get the file paths in the current folder
    import os
    paths = []
    list = os.listdir(folder)
    for fname in list:
        if os.path.splitext(fname)[-1] == fext:
            paths.append(os.path.join(folder, fname))

    paths.sort()
    return paths


def get_all_paths(folder, fext):
    # Get all file paths (include subfolders) in the current folder
    import os
    paths = []
    for home, dirs, files in os.walk(folder):
        for fname in files:
            if os.path.splitext(fname)[-1] == fext:
                paths.append(os.path.join(home, fname))
    paths.sort()
    return paths


def join(path, item):
    return os.path.join(path, item).replace("\\", "/")


# make dir if not exist
def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


# clear dir if exist
def clean_dir_if_exist(path):
    if os.path.exists(path):
        shutil.rmtree(path)


# move all files to its parent path
def move_file_to_father_dir(sparse_path):
    scenes = os.listdir(sparse_path)
    cnt = 0
    for s in scenes:
        files_list = os.listdir(join(sparse_path, s))
        for files in files_list:
            if cnt == 0:
                shutil.copy(join(files_list, files), join(sparse_path, files))
            else:
                shutil.copy(join(files_list, files), join(sparse_path, s + files))
        cnt = cnt + 1


# make father dir if not exist
def mk_father_dir_if_not_exist(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.mkdir(dir)


def get_file_path_and_name(path):
    name = os.path.basename(path)
    father_dir = os.path.dirname(path)
    dir, father_name = os.path.split(father_dir)

    return name, father_name, dir


def find_file_with_ext(root_dir, select_file_paths, ext):
    files = os.listdir(root_dir)
    for f in files:
        fl = os.path.join(root_dir, f)
        if os.path.isdir(fl):
            find_file_with_ext(fl, select_file_paths, ext)
        if os.path.isfile(fl) and os.path.splitext(fl)[1] == ext:
            select_file_paths.append(fl)

    return select_file_paths






