#!/usr/bin/env python
# coding=UTF-8
"""
Author: Wei Luo
Date: 2025-03-10 22:20:30
LastEditors: Wei Luo
LastEditTime: 2025-03-10 23:04:15
Note: Note
"""

import casadi as ca
import numpy as np

pos = ca.MX.sym("pos", 2)
theta = ca.MX.sym("theta")
v = ca.MX.sym("v")
omega = ca.MX.sym("omega")

# States
state = ca.vertcat(pos, theta)

# Controls
control = ca.vertcat(v, omega)

# ODE rhs
ode = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)

# discretize system
dt = 0.1
sys = {}
sys["x"] = state
sys["u"] = control
sys["ode"] = ode * dt

intg = ca.integrator(
    "intg", "rk", sys, 0, 1, {"simplify": True, "number_of_finite_elements": 4}
)

F = ca.Function(
    "F",
    [state, control],
    [intg(x0=state, u=control)["xf"]],
    ["x", "u"],
    ["xnext"],
)

nx = state.numel()
nu = control.numel()

f = 0  # Objective
x = []  # all decision variable symbols
lbx = []
ubx = []
x0 = []
g = []
lbg = []
ubg = []
p = []
equality = []

N = 20
X = [ca.MX.sym("X", nx) for i in range(N + 1)]
x += X
for k in range(N + 1):
    x0.append(ca.vertcat(0, 0, 0))
    lbx.append(ca.vertcat(-3.0, -3.0, -np.pi))
    ubx.append(ca.vertcat(3.0, 3.0, np.pi))
U = [ca.MX.sym("U", nu) for i in range(N)]
x += U

for k in range(N):
    x0.append(ca.vertcat(0, 0))
    lbx.append(-ca.DM.inf(nx, 1))
    ubx.append(ca.DM.inf(nx, 1))
