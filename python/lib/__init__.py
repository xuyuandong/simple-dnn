#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""lib is a python package

when use absolute import in package, can not execute file directly (ImportError)

add package dir to sys.path fix this issue.

cd directory outside lib, then 

python -m lib.dataset

"""
import os
import sys
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)
