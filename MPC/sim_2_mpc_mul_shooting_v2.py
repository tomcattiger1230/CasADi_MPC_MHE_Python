#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
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

    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    states = ca.vertcat(x, y)
    states = ca.vertcat(states, theta)
    n_states = states.size()[0]

    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]

    ## rhs
    rhs = ca.vertcat(v*np.cos(theta), v*np.sin(theta))
    rhs = ca.vertcat(rhs, omega)

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', n_controls, N)

    X = ca.SX.sym('X', n_states, N+1)

    P = ca.SX.sym('P', n_states+n_states)


    ### define
    

    #### define the relationship within the horizon
    # for i in range(N):
    #     f_value = f(X[:, i], U[:, i])
    #     X[:, i+1] = X[:, i] + f_value*T

    # ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

    Q = np.array([[1.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, .1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    g = [] # equal constrains
    for i in range(N):
        obj = obj + ca.mtimes([(X[:, i]-P[3:]).T, Q, X[:, i]-P[3:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])
        x_next_ = f(X[:, i], U[:, i])*T +X[:, i]
        g.append(X[:, i+1]-x_next_)

    opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

    nlp_prob = {'f': obj, 'x': opt_variables, 'p':P, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    lbg = 0.0
    ubg = 0.0
    lbx = []
    ubx = []
    for _ in range(N):
        lbx.append(-v_max)
        lbx.append(-omega_max)
        ubx.append(v_max)
        ubx.append(omega_max)
        lbx.append(-2.0)
        lbx.append(-2.0)
        lbx.append(-np.inf)
        ubx.append(2.0)
        ubx.append(2.0)
        ubx.append(np.inf)
    # for the N+1 state
    lbx.append(-2.0)
    lbx.append(-2.0)
    lbx.append(-np.inf)
    ubx.append(2.0)
    ubx.append(2.0)
    ubx.append(np.inf)

    # Simulation
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)# initial state
    x0_ = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)
    x_m = np.zeros((n_states, N+1))
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
    # c_p = P(0)
    #init_control = U(0)
    # print(u0.shape) u0 should have (n_controls, N)
    while(np.linalg.norm(x0-xs)>1e-2 and mpciter-sim_time/T<0.0 ):
        ## set parameter
        # p_ = np.concatenate((x0, xs))
        c_p = np.concatenate((x0, xs))
        init_control = np.concatenate((u0.reshape(-1, 1), x_m.reshape(-1, 1)))
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        estimated_opt = res['x'].full() # the feedback is in the series [u0, x0, u1, x1, ...]
        print(estimated_opt)
        u0 = estimated_opt[:200].reshape(n_controls, N)
        print(u0.T)
        ff_value = ff(u0, c_p) # [n_states, N]
        print(ff_value.T)
        x_c.append(ff_value)
        u_c.append(u0[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u0, f)
        # x0 = x0.toarray().reshape(-1, 1)
        # u0 = np.array([1,2]*N).reshape(-1, 2).T# np.ones((N, 2)) # controls
        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        mpciter = mpciter + 1

    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0_, target_state=xs, robot_states=xx )
