# -*- coding: utf-8 -*-
#   ___      _ _                    _     
#  / _ \    | | |                  | |    
# / /_\ \ __| | |__   ___  ___  ___| |__  
# |  _  |/ _` | '_ \ / _ \/ _ \/ __| '_ \ 
# | | | | (_| | | | |  __/  __/\__ \ | | |
# \_| |_/\__,_|_| |_|\___|\___||___/_| |_|
# Date:   2021-03-18 00:07:49
# Last Modified time: 2021-03-18 00:13:45

import json
import os

def get_class_id(labels):
	filename="Dataset/imagenet_class_index.json"

	with open(filename) as f:
		imagenet_classes= {labels[1]: int(idx) for (idx, labels) in
			json.load(f).items()}
	return imagenet_classes[labels]