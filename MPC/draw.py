#!/usr/bin/env python
# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def Draw_MPC_point_stabilization_v1(robot_states, target_state, rob_diam=0.3):
    radius_r = rob_diam/2. # radius of the robot
    fig, ax = plt.subplots()

    # plot target state
    target_circle = plt.Circle(target_state[:2], radius_r, color='b', fill=False)
    ax.add_artist(target_circle)
    target_arr = mpatches.Arrow(target_state[0], target_state[1], radius_r*np.cos(target_state[2]), radius_r*np.sin(target_state[2]), width=0.2)
    ax.add_patch(target_arr)
    ax.add_artist(target_circle)
    for i in range(len(robot_states)):
        plt.clf()
        target_circle = plt.Circle(target_state[:2], radius_r, color='b', fill=False)
        ax.add_artist(target_circle)
        target_arr = mpatches.Arrow(target_state[0], target_state[1], radius_r*np.cos(target_state[2]), radius_r*np.sin(target_state[2]), width=0.2)
        ax.add_patch(target_arr)
        ax.add_artist(target_circle)
        position = robot_states[i][:2]
        orientation = robot_states[i][2]
        robot_body = plt.Circle(position, radius_r, color='r', fill=False)

        ax.add_artist(robot_body)
        robot_arr = mpatches.Arrow(position[0], position[1], radius_r*np.cos(orientation), radius_r*np.sin(orientation), width=0.2, color='r')
        ax.add_patch(robot_arr)
        plt.pause(0.1)


        plt.xlim(-0.8, 2.0)
        plt.ylim(-0.8, 2.0)
        plt.grid('--')
    # plt.axis('equal')
        plt.show()