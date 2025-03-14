#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca

import numpy as np
import time
from draw import Draw_MPC_point_stabilization_v1


def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f(x0, u[0])
    st = x0 + T * f_value.full().squeeze()
    t = t0 + T
    u_end = np.concatenate((u[1:], u[-1:]), axis=0)
    x_f = np.concatenate((x_f[1:], x_f[-1:]), axis=0)

    return t, st, u_end, x_f


def init_guess_array(init_x: np.array, init_u: np.array, N: int):
    x_guess = []
    if init_x.shape[1] != 1:
        for i in range(N + 1):
            x_guess.append(ca.vcat(init_x[i]))
            if i < N:
                x_guess.append(ca.vcat(init_u[i]))
    else:
        print("the size is wrong {}".format(init_x.shape))
    return x_guess


def result_output(result_array: np.array, state_size: int, control_size: int, N: int):
    state_result = np.zeros((N + 1, state_size))
    control_result = np.zeros((N, control_size))
    result_ = result_array[:-state_size].reshape(-1, state_size + control_size)
    for i in range(int(result_array.shape[0] // (state_size + control_size))):
        state_result[i] = result_[i, :state_size]
        control_result[i] = result_[i, state_size:]
    state_result[-1] = result_array[-state_size:].reshape(-1, state_size)

    return np.array(state_result), np.array(control_result)


if __name__ == "__main__":
    T = 0.2  # sampling time [s]
    N = 100  # prediction horizon
    rob_diam = 0.3  # [m]
    v_max = 0.6
    omega_max = np.pi / 4.0

    x = ca.MX.sym("x")
    y = ca.MX.sym("y")
    theta = ca.MX.sym("theta")
    states = ca.vertcat(x, y)
    states = ca.vertcat(states, theta)
    n_states = states.numel()

    v = ca.MX.sym("v")
    omega = ca.MX.sym("omega")
    controls = ca.vertcat(v, omega)
    n_controls = controls.numel()

    # rhs
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta))
    rhs = ca.vertcat(rhs, omega)

    # discretize system
    # function
    F = ca.Function(
        "F", [states, controls], [rhs], ["input_state", "control_input"], ["rhs"]
    )

    # for MPC
    # define
    Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    # cost function
    obj = 0  # cost
    g = []  # equal constrains
    lbg = []  # lower bound of constraints
    ubg = []  # upper bound of constraints
    equality_list = []
    X = []  # list of all states
    U = []  # list of all controls
    sym_symbol_list = []
    lbx = []
    ubx = []
    parameter_list = []

    # initial constraints
    Xinit = ca.MX.sym("X0", n_states)
    parameter_list.append(Xinit)
    X_expected = ca.MX.sym("X_exp", n_states)
    parameter_list.append(X_expected)

    # state definitions according to Fatrop's rule
    for i in range(N + 1):
        sym = ca.MX.sym("x", n_states)
        sym_symbol_list.append(sym)
        X.append(sym)
        lbx.append(-2.0)
        lbx.append(-2.0)
        lbx.append(-np.inf)
        ubx.append(2.0)
        ubx.append(2.0)
        ubx.append(np.inf)
        if i < N:
            sym = ca.MX.sym("u", n_controls)
            sym_symbol_list.append(sym)
            U.append(sym)
            lbx.append(-v_max)
            ubx.append(v_max)
            lbx.append(-omega_max)
            ubx.append(omega_max)
    # system dynamics constraints
    for i in range(N):
        g.append(X[i + 1] - X[i] - F(X[i], U[i]) * T)
        lbg.append(ca.DM.zeros(n_states, 1))
        ubg.append(ca.DM.zeros(n_states, 1))
        equality_list += [True] * n_states
        if i == 0:
            g.append(X[0] - Xinit)
            lbg.append(ca.DM.zeros(n_states, 1))
            ubg.append(ca.DM.zeros(n_states, 1))
            equality_list += [True] * n_states

    # X_expected = ca.vertcat(1.5, 1.5, 0.0)
    for i in range(N):
        obj = (
            obj
            + ca.mtimes([(X[i] - X_expected).T, Q, X[i] - X_expected])
            + ca.mtimes([U[i].T, R, U[i]])
        )

    # Solve the problem
    nlp = {}
    nlp["f"] = obj
    nlp["g"] = ca.vcat(g)
    nlp["x"] = ca.vcat(sym_symbol_list)
    nlp["p"] = ca.vcat(parameter_list)

    options = {}
    options["expand"] = False
    options["fatrop"] = {"mu_init": 0.1}
    options["structure_detection"] = "auto"
    options["debug"] = False
    options["equality"] = equality_list

    solver = ca.nlpsol("solver", "fatrop", nlp, options)

    # Simulation
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0])  # initial state
    x_begin = x0.copy()
    # x_m = np.zeros((n_states, N + 1))
    x_m = np.zeros((N + 1, n_states))
    next_states = x_m.copy()
    xs = np.array([1.5, 1.5, 0.0])  # final state
    u0 = np.array([1, 2] * N).reshape(-1, 2)  # np.ones((N, 2)) # controls
    x_c = []  # contains for the history of the state
    u_c = []
    t_c = [t0]  # for the time
    xx = []
    sim_time = 20.0

    # start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    # initial test
    while np.linalg.norm(x0 - xs) > 1e-2 and mpciter - sim_time / T < 0.0:
        # set parameter
        c_p = []
        c_p.append(ca.vcat(x0))
        c_p.append(ca.vcat(xs))
        init_guess = init_guess_array(next_states, u0, N)
        t_ = time.time()
        res = solver(
            x0=ca.vcat(init_guess),
            p=ca.vcat(c_p),
            lbg=ca.vcat(lbg),
            lbx=ca.vcat(lbx),
            ubg=ca.vcat(ubg),
            ubx=ca.vcat(ubx),
        )
        index_t.append(time.time() - t_)
        estimated_opt = res["x"].full()  #
        x_result, u_result = result_output(estimated_opt, n_states, n_controls, N)
        u0 = u_result.copy()
        x_c.append(x_result.T)
        u_c.append(u_result[:, 0])
        t_c.append(t0)
        t0, x0, u0, next_states = shift_movement(T, t0, x0, u0, x_m, F)
        # x0 = ca.reshape(x0, -1, 1)
        # x0 = x0.full()
        xx.append(x0)
        mpciter = mpciter + 1
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time) / (mpciter))

    draw_result = Draw_MPC_point_stabilization_v1(
        rob_diam=0.3, init_state=x_begin, target_state=xs, robot_states=xx
    )
