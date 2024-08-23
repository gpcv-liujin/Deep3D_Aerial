#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2024, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu


import sys
import os
import time

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def Save_Logger(log_path):

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_file_name = log_path + '/log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime())+'.log'
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)


