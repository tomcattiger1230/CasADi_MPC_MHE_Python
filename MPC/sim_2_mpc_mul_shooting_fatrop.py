#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
import time
from draw import Draw_MPC_point_stabilization_v1


def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T * f_value.full()
    t = t0 + T
    u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)

    return t, st, u_end, x_f


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
    n_states = states.size()[0]

    v = ca.MX.sym("v")
    omega = ca.MX.sym("omega")
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]

    # rhs
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta))
    rhs = ca.vertcat(rhs, omega)

    # discreteize system
    # dt = ca.MX.sym("dt")
    sys_dyn = {}
    sys_dyn["x"] = states
    sys_dyn["u"] = controls
    # sys_dyn["p"] = dt
    sys_dyn["ode"] = rhs * T
    print(rhs * T)

    intg = ca.integrator(
        "intg", "rk", sys_dyn, 0, 1, {"simplify": True, "number_of_finite_elements": 3}
    )

    F = ca.Function(
        "F",
        [states, controls],
        [intg(x0=states, u=controls)["xf"], ["x", "u"], ["x_next"]],
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
    Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    # cost function
    obj = 0  #### cost
    g = []  # equal constrains
    equality_list = []
    g.append(X[:, 0] - P[:3])
    equality_list += [True] * 3

    for i in range(N):
        obj = (
            obj
            + ca.mtimes([(X[:, i] - P[3:]).T, Q, X[:, i] - P[3:]])
            + ca.mtimes([U[:, i].T, R, U[:, i]])
        )
        # x_next_ = f(X[:, i], U[:, i]) * T + X[:, i]
        # g.append(X[:, i + 1] - x_next_)
        g.append(X[:, i + 1] - F(X[:, i], U[:, i]))
        equality_list += [True] * n_states

    opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

    nlp_prob = {"f": obj, "x": opt_variables, "p": P, "g": ca.vertcat(*g)}
    opts_setting = {}
    opts_setting["expand"] = True
    opts_setting["fatrop"] = {"mu_init": 0.1}
    opts_setting["structure_detection"] = "auto"
    opts_setting["debug"] = True
    opts_setting["equality"] = equality_list

    solver = ca.nlpsol("solver", "fatrop", nlp_prob, opts_setting)

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
        lbx.append(-2.0)
        lbx.append(-2.0)
        lbx.append(-np.inf)
        ubx.append(2.0)
        ubx.append(2.0)
        ubx.append(np.inf)

    # Simulation
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)  # initial state
    x0_ = x0.copy()
    x_m = np.zeros((n_states, N + 1))
    next_states = x_m.copy()
    xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1)  # final state
    u0 = np.array([1, 2] * N).reshape(-1, 2).T  # np.ones((N, 2)) # controls
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
        c_p = np.concatenate((x0, xs))
        init_control = np.concatenate(
            (u0.T.reshape(-1, 1), next_states.T.reshape(-1, 1))
        )
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time() - t_)
        estimated_opt = res[
            "x"
        ].full()  # the feedback is in the series [u0, x0, u1, x1, ...]
        u0 = estimated_opt[:200].reshape(N, n_controls).T  # (n_controls, N)
        x_m = estimated_opt[200:].reshape(N + 1, n_states).T  # [n_states, N]
        x_c.append(x_m.T)
        u_c.append(u0[:, 0])
        t_c.append(t0)
        t0, x0, u0, next_states = shift_movement(T, t0, x0, u0, x_m, f)
        x0 = ca.reshape(x0, -1, 1)
        x0 = x0.full()
        xx.append(x0)
        mpciter = mpciter + 1
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time) / (mpciter))

    draw_result = Draw_MPC_point_stabilization_v1(
        rob_diam=0.3, init_state=x0_, target_state=xs, robot_states=xx
    )
