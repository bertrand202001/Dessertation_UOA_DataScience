#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:48:14 2020

@author: yujzhang
"""

import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import glob

path = '/Users/yujzhang/Desktop' 
all_files = glob.glob(path + "/adult3.csv")
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

print(frame.describe())
    
profile = ProfileReport(frame, title="Pandas Profiling Report")

profile.to_file("/Users/yujzhang/Downloads/your_report.html")
