#!/usr/bin/env python
# coding=utf-8

import casadi as ca 
import numpy as np


if __name__ == '__main__':
    T = 0.2
    N = 100
    v_max = 0.6
    omega_max = np.pi/4.0

    opti = ca.Opti()
    # control variables, linear velocity v and angle velocity omega
    controls = opti.variable(2, N)
    v = controls[0, :]
    omega = controls[1, :]
    states = opti.variable(3, h)
    x = states[0, :]
    y = states[1, :]
    theta = states[2, :]
  
