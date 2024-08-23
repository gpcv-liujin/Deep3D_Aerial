# Copyright (c) 2024, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu


import json
import sys


def parseArguments(config_path):

	with open(config_path, encoding='utf-8') as json_file:
		tp = json.load(json_file)
	return tp
