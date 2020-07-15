#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
import time
from draw import Draw_MPC_point_stabilization_v1

def shift_movement(T, t0, x0, u, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T*f_value
    t = t0 + T
    u_end = ca.horzcat(u[:, 1:], u[:, -1])

    return t, st, u_end

if __name__ == '__main__':
    T = 0.2 # sampling time [s]
    N = 100 # prediction horizon
    rob_diam = 0.3 # [m]
    v_max = 0.6
    omega_max = np.pi/4.0

    states = ca_tools.struct_symSX([
        (
            ca_tools.entry('x'),
            ca_tools.entry('y'),
            ca_tools.entry('theta')
        )
    ])
    x, y, theta = states[...]
    n_states = states.size

    controls  = ca_tools.struct_symSX([
        (
            ca_tools.entry('v'),
            ca_tools.entry('omega')
        )
    ])
    v, omega = controls[...]
    n_controls = controls.size

    ## rhs
    rhs = ca_tools.struct_SX(states)
    rhs['x'] = v*ca.cos(theta)
    rhs['y'] = v*ca.sin(theta)
    rhs['theta'] = omega
    # rhs = v*np.cos(theta)
    # rhs = ca.vertcat(rhs, v*np.sin(theta))
    # rhs = ca.vertcat(rhs, omega)

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    optimizing_target = ca_tools.struct_symSX([
        (
            ca_tools.entry('U', repeat=N, struct=controls)
        )
    ])
    U, = optimizing_target[...] # data are stored in list [], notice that ',' cannot be missed

    X = ca.SX.sym('X', n_states, N+1)

    current_parameters = ca_tools.struct_symSX([
        (
            ca_tools.entry('P', shape=n_states+n_states),
        )
    ])
    P, = current_parameters[...]
    #P = ca.SX.sym('P', n_states+n_states) # initial pose and final pose

    ### define
    X[:, 0] = P[:3] # initial condiction

    #### define the relationship within the horizon
    for i in range(N):
        f_value = f(X[:, i], U[i])
        X[:, i+1] = X[:, i] + f_value*T

    ff = ca.Function('ff', [optimizing_target, current_parameters], [X], ['input_U', 'target_state'], ['horizon_states'])

    Q = np.array([[1.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, .1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    for i in range(N):
        obj = obj + ca.mtimes([(X[:, i]-P[3:]).T, Q, X[:, i]-P[3:]]) + ca.mtimes([U[i].T, R, U[i]])

    #### constrains
    g = [] # equal constrains
    for i in range(N+1):
        g.append(X[0, i])
        g.append(X[1, i])

    nlp_prob = {'f': obj, 'x': optimizing_target, 'p':current_parameters, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    lbg = -2.0
    ubg = 2.0
    lbx = []
    ubx = []
    for _ in range(N):
        lbx.append(-v_max)
        lbx.append(-omega_max)
        ubx.append(v_max)
        ubx.append(omega_max)

    # Simulation
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)# initial state
    xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1) # final state
    u0 = np.array([1,2]*N).reshape(-1, 2).T# np.ones((N, 2)) # controls
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0
    ### inital test
    c_p = current_parameters(0)
    init_control = optimizing_target(0)
    start_time = time.time()
    index_t = []
    while(np.linalg.norm(x0-xs)>1e-2 and mpciter-sim_time/T<0.0 ):
        ## set parameter
        # p_ = np.concatenate((x0, xs))
        c_p['P'] = np.concatenate((x0, xs))
        init_control['U', lambda x:ca.horzcat(*x)] = u0 #[:, 0:N]
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time()- t_)
        u0 = ca.reshape(res['x'], n_controls, N)
        ff_value = ff(init_control, c_p) # [n_states, N]
        x_c.append(ff_value)
        u_c.append(u0[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u0, f)
        # x0 = x0.toarray().reshape(-1, 1)
        # u0 = np.array([1,2]*N).reshape(-1, 2).T# np.ones((N, 2)) # controls
        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        mpciter = mpciter + 1
    t_v = np.array(index_t)
    print(t_v.mean()) 
    print((time.time() - start_time)/(mpciter))

    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0.full(), target_state=xs, robot_states=xx )
