#!/usr/bin/env python
# coding=utf-8

import casadi as ca
import numpy as np


def prediction_function(T, N):
    # define predition horizon function
    states = ca.SX.sym('states', N+1, 3)
    x0 = ca.SX.sym('x0', 3)
    u = ca.SX.sym('u', N, 2)
    states[0, :] = x0
    for i in range(N):
        states[i+1, 0] = states[i, 0] + u[i, 0] * np.cos(states[i, 2]) * T
        states[i+1, 1] = states[i, 1] + u[i, 0] * np.sin(states[i, 2]) * T
        states[i+1, 2] = states[i, 2] + u[i, 1] * T
    func = ca.Function('ff', [x0, u], [states], ['init_s', 'controls_horizon'], ['outputs'])

    return func


# def prediction(x0, u, T, N):
#    # define predition horizon function
#    states = ca.MX.sym('state', N+1, 3)
#    states[0, :] = x0
#    x_ = states[:, 0]
#    y_ = states[:, 1]
#    theta_ = states[:, 2]
#    v_ = u[:, 0]
#    omega_ = u[:, 1]
#    for i in range(N):
#        x_[i+1] = x_[i] + v_[i] * np.cos(theta_[i]) * T
#        y_[i+1] = y_[i] + v_[i] * np.sin(theta_[i]) * T
#        theta_[i+1] = theta_[i] + omega_[i] * T
#    return states

if __name__ == '__main__':
    T = 0.2
    N = 100
    v_max = 0.6
    omega_max = np.pi/4.0

    opti = ca.Opti()
    # control variables, linear velocity v and angle velocity omega
    controls = opti.variable(N, 2)
    v = controls[:, 0]
    omega = controls[:, 1]
    states = opti.variable(N+1, 3)
    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]

    # parameters
    x0 = opti.parameter(3)
    xs = opti.parameter(3)
    # create model
    f = lambda x, u: ca.vertcat(*[u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])
    ## prediction function
    ff = prediction_function(T, N)
    print(ff)
    ## init_condition
    states[0, :] = x0
    for i in range(N):
        x_next = states[i, :] + f(states[i, :], controls[i, :]).T*T
        opti.subject_to(states[i+1, :]==x_next)

    ## define the cost function
    ### some addition parameters
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, .1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    for i in range(N):
        obj = obj + ca.mtimes([(states[i, :]-xs.T), Q, (states[i, :]-xs.T).T]) + ca.mtimes([controls[i, :], R, controls[i, :].T])

    opti.minimize(obj)

    #### boundrary and control conditions
    opti.subject_to(opti.bounded(-2.0, x, 2.0))
    opti.subject_to(opti.bounded(-2.0, y, 2.0))
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))
    # opti.subject_to(-2.0<x)
    #opti.subject_to(x<2.0)
    #opti.subject_to(y>-2.0)
    #opti.subject_to(y<2.0)
    #opti.subject_to(v<v_max)
    #opti.subject_to(v>-v_max)
    #opti.subject_to(omega<omega_max)
    #opti.subject_to(omega>-omega_max)

    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    opti.solver('ipopt', opts_setting)
    final_state = np.array([1.5, 1.5, 0.0])
    opti.set_value(xs, final_state)

    t0 = 0
    init_state = np.array([0.0, 0.0, 0.0])
    current_state = init_state.copy()
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0

    while(np.linalg.norm(current_state-final_state)>1e-2 and mpciter-sim_time/T<0.0 and mpciter<2 ):
        ## set parameter
        opti.set_value(x0, current_state)
        sol = opti.solve()
        u = sol.value(controls)
        next_states = ff(ca.DM([0.0, 0.0, 0.0]), u)
        print(next_states)
        mpciter = mpciter + 1
