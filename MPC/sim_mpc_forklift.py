#!/usr/bin/env python
# coding=UTF-8
"""
Author: Wei Luo
Date: 2023-05-30 21:34:39
LastEditors: Wei Luo
LastEditTime: 2023-05-31 10:35:57
Note: Note
"""

import casadi as ca
from draw import Draw_FolkLift

import numpy as np
import time


# define a movement at next time step
def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T * f_value.full()
    t = t0 + T
    u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)

    return t, st, u_end, x_f


if __name__ == "__main__":
    T = 0.2
    N = 100
    l = 1.0
    v_max = 0.6
    omega_max = np.pi / 4.0

    x = ca.MX.sym("x")
    y = ca.MX.sym("y")
    theta = ca.MX.sym("theta")
    alpha = ca.MX.sym("alpha")
    states = ca.vertcat(x, y, theta, alpha)
    n_states = states.size()[0]

    v = ca.MX.sym("v")
    omega = ca.MX.sym("omega")
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]

    # rhs
    rhs = ca.vertcat(
        v * ca.cos(theta) * ca.cos(alpha),
        v * ca.sin(theta) * ca.cos(alpha),
        v / l * ca.sin(alpha),
        omega,
    )

    # function
    f = ca.Function(
        "f", [states, controls], [rhs], ["input_state", "control_input"], ["rhs"]
    )

    # for MPC
    U = ca.MX.sym("U", n_controls, N)

    X = ca.MX.sym("X", n_states, N + 1)

    P = ca.MX.sym("P", n_states + n_states)

    # define
    Q = np.array(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 2., 0.0],
            [0.0, 0.0, 0.0, 0.1],
        ]
    )
    R = np.array([[0.5, 0.0], [0.0, .4]])
    # cost function
    obj = 0  # cost
    g = []  # equal constrains
    g.append(X[:, 0] - P[:n_states])

    for i in range(N):
        obj = (
            obj
            + ca.mtimes([(X[:, i] - P[n_states:]).T, Q, X[:, i] - P[n_states:]])
            + ca.mtimes([U[:, i].T, R, U[:, i]])
        )
        x_next_ = f(X[:, i], U[:, i]) * T + X[:, i]
        g.append(X[:, i + 1] - x_next_)

    opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

    nlp_prob = {"f": obj, "x": opt_variables, "p": P, "g": ca.vertcat(*g)}
    opts_setting = {
        "ipopt.max_iter": 100,
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.acceptable_tol": 1e-8,
        "ipopt.acceptable_obj_change_tol": 1e-6,
    }

    solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts_setting)

    lbg = 0.0
    ubg = 0.0
    lbx = []
    ubx = []
    for _ in range(N):
        lbx.append(-v_max)
        lbx.append(-omega_max)
        ubx.append(v_max)
        ubx.append(omega_max)
    for _ in range(
        N + 1
    ):  # note that this is different with the method using structure
        lbx.append(-16.0)
        lbx.append(-16.0)
        lbx.append(-np.pi)
        lbx.append(-np.pi / 2.0)
        ubx.append(7.01)
        ubx.append(7.01)
        ubx.append(np.pi)
        ubx.append(np.pi / 2.0)

    # simulation
    t0 = 0.0
    x_init = np.array([0.0, 0.0, 0.0, 0.0])
    x_current = x_init.copy().reshape(-1, 1)
    x_target = np.array([7, 7, 0.0, 0.0]).reshape(-1, 1)
    u_guess = np.array([0.0, 0.0] * N).reshape(-1, 2)
    x_guess = np.zeros((n_states, N + 1))
    start_time = time.time()
    cal_time_list = []
    state_results = []
    control_results = []
    time_step_list = []
    final_state_results = []
    mpc_iter = 0
    while np.linalg.norm(x_target - x_init) > 1e-3 and mpc_iter < 100:
        # parameters
        c_p = np.concatenate((x_current, x_target))
        # guessing optimization states
        init_opt_state = np.concatenate(
            (u_guess.T.reshape(-1, 1), x_guess.T.reshape(-1, 1))
        )
        t_ = time.time()
        opt_result = solver(
            x0=init_opt_state, p=c_p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx
        )
        cal_time_list.append(time.time() - t_)
        estimated_result = opt_result["x"].full()
        u_guess = estimated_result[: n_controls * N].reshape(N, n_controls).T
        x_guess = estimated_result[n_controls * N :].reshape(N + 1, n_states).T
        # print(x_guess.T)
        state_results.append(x_guess.T)
        final_state_results.append(x_guess.T[0])
        control_results.append(u_guess[:, 0])
        time_step_list.append(t0)
        t0, x_current, u_guess, x_guess = shift_movement(
            T, t0, x_current, u_guess, x_guess, f
        )
        mpc_iter += 1

    Draw_FolkLift(final_state_results, x_init, False)
