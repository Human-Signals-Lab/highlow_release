# -*- coding: utf-8 -*-
"""
Created on Fri Jun 7 2024
"""

import os

def check_dir(rootdir):
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)