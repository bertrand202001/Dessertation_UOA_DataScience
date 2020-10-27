#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:11:39 2019

@author: yujzhang
"""

import numpy as np
import matplotlib.pyplot as plt

r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r

ax = plt.subplot(111, projection='polar')
ax.plot(theta, r)
ax.set_rmax(2)
ax.set_rticks([])
ticks= np.linspace(0,360,9)[:-1] 
ax.set_xticks(np.deg2rad(ticks))
ticklabels = ["".join(np.random.choice(list("ABCDE"),size=15)) for _ in range(len(ticks))]
ax.set_xticklabels(ticklabels, fontsize=10)

plt.gcf().canvas.draw()
angles = np.linspace(0,2*np.pi,len(ax.get_xticklabels())+1)
angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
angles = np.rad2deg(angles)
labels = []
for label, angle in zip(ax.get_xticklabels(), angles):
    x,y = label.get_position()
    lab = ax.text(x,y-.65, label.get_text(), transform=label.get_transform(),
                  ha=label.get_ha(), va=label.get_va())
    lab.set_rotation(angle)
    labels.append(lab)
ax.set_xticklabels([])

plt.subplots_adjust(top=0.68,bottom=0.32,left=0.05,right=0.95)
plt.show()