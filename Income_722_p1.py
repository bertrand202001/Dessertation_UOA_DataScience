#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 16:38:24 2019

@author: yujzhang
"""

import pandas as pd
import glob
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import numpy as np
import sklearn.tree as st
import seaborn as sns
import matplotlib.ticker as ticker


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12


path = '.' 
all_files = glob.glob(path + "/*.csv")
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

print(frame.describe())


#Continus Variables
age = frame['age']
education_num = frame['education-num']
capital_gain = frame['capital-gain']
capital_loss = frame['capital-loss']
hours_per_week = frame['hours-per-week']

#Category Variables
workclass = frame['workclass']
marital_status = frame['marital-status']
occupation = frame['occupation']
relationship = frame['relationship']
race = frame['race']
sex = frame['sex']
native_country = frame['native-country']
income = frame['income']

column_list = ['age', 'education-num','capital-gain', 'capital-loss', 
               'hours-per-week']

cat_col_list = ['workclass', 'education', 'marital-status', 'occupation', 
                'relationship', 'race', 'sex','native-country','income']

col_list = frame.columns

#=================Raw Data Visualization======================================

f, (ax1) = mp.subplots(figsize=(8, 8))
ax1 = sns.distplot(age, rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE of age"},
                   hist_kws={"histtype": "step", "linewidth": 3,
                             "alpha": 1, "color": "g", "label":"Histgram of age"})


f, (ax2) = mp.subplots(figsize=(8, 8))    
ax2 = sns.distplot(education_num, rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE of education_num"},
                   hist_kws={"histtype": "step", "linewidth": 3,
                             "alpha": 1, "color": "g", "label":"Histgram of education_num"})


f, (ax3) = mp.subplots(figsize=(8, 8))    
ax3 = sns.distplot(capital_gain, rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE of capital_gain"},
                   hist_kws={"histtype": "step", "linewidth": 3,
                             "alpha": 1, "color": "g", "label":"Histgram of capital_gain"})
    
f, (ax4) = mp.subplots(figsize=(8, 8))    
ax4 = sns.distplot(capital_loss, rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE of capital_loss"},
                   hist_kws={"histtype": "step", "linewidth": 3,
                             "alpha": 1, "color": "g", "label":"Histgram of capital_loss"})
    
f, (ax5) = mp.subplots(figsize=(8, 8))
ax5 = sns.distplot(hours_per_week, rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE of hours_per_week"},
                   hist_kws={"histtype": "step", "linewidth": 3,
                             "alpha": 1, "color": "g", "label":"Histgram of hours_per_week"})

for cal in cat_col_list:
    mp.figure(figsize=(20,20))
    ax7 = sns.countplot(x=cal, data=frame, palette="ch:.25")
    mp.title('Distribution of ' + cal)
    mp.xlabel('Number of ' + cal)
    mp.xticks(rotation=45)
    
    # Make twin axis
    ax8=ax7.twinx()
    
    # Switch so count axis is on right, frequency on left
    ax8.yaxis.tick_left()
    ax7.yaxis.tick_right()
    
    # Also switch the labels over
    ax7.yaxis.set_label_position('right')
    ax8.yaxis.set_label_position('left')
    
    ax8.set_ylabel('Frequency [%]')
    
    
    for p in ax7.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax7.annotate('{:.1f}%'.format(100.*y/len(frame[cal])), (x.mean(), y), ha='center', va='bottom') # set the alignment of the text
    
    # Use a LinearLocator to ensure the correct number of ticks
    ax7.yaxis.set_major_locator(ticker.LinearLocator(11))
    
    # Fix the frequency range to 0-100
    ax8.set_ylim(0,100)
    ax7.set_ylim(0,len(frame[cal]))
    
    # And use a MultipleLocator to ensure a tick spacing of 10
    ax8.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    ax8.grid(None)
    
    mp.rcParams.update({'font.size': 18})
    
    mp.savefig('/Users/yujzhang/Documents/722_plot/plot'+cal+'.png', format='png')
    
    
