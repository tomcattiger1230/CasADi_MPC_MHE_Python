#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import numpy as np


if __name__ == '__main__':
    T = 0.2 
    N = 10
    rob_diam = 0.3 # [m]
    v_max = 0.6
    omega_max = np.pi/4.0

    