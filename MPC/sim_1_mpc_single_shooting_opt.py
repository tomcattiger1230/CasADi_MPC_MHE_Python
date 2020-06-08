#!/usr/bin/env python
# coding=utf-8

import casadi as ca
import numpy as np

from draw import Draw_MPC_point_stabilization_v1

def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f(x0, u[0]) # u in shape (N, 2)
    st = x0 + T*f_value
    t = t0 + T
    u_end = np.concatenate((u[1:], u[-1:]))
    x_f = np.concatenate((x_f[1:], x_f[-1:]))

    return t, st, u_end, x_f

def prediction_function(T, N):
    # define predition horizon function
    states = ca.MX.sym('states', N+1, 3)
    x0 = ca.MX.sym('x0', 3)
    u = ca.MX.sym('u', N, 2)
    states[0, :] = x0
    for i in range(N):
        states[i+1, 0] = states[i, 0] + u[i, 0] * np.cos(states[i, 2]) * T
        states[i+1, 1] = states[i, 1] + u[i, 0] * np.sin(states[i, 2]) * T
        states[i+1, 2] = states[i, 2] + u[i, 1] * T
    func = ca.Function('ff', [x0, u], [states], ['init_s', 'controls_horizon'], ['outputs'])

    return func


def prediction_state(x0, u, T, N):
    # define predition horizon function
    states_ = np.zeros((N+1, 3))
    states_[0, :] = x0
    for i in range(N):
        states_[i+1, 0] = states_[i, 0] + u[i, 0] * np.cos(states_[i, 2]) * T
        states_[i+1, 1] = states_[i, 1] + u[i, 0] * np.sin(states_[i, 2]) * T
        states_[i+1, 2] = states_[i, 2] + u[i, 1] * T
    return states_

if __name__ == '__main__':
    T = 0.2
    N = 100
    v_max = 0.6
    omega_max = np.pi/4.0

    opti = ca.Opti()
    # control variables, linear velocity v and angle velocity omega
    opt_controls = opti.variable(N, 2)
    v = opt_controls[:, 0]
    omega = opt_controls[:, 1]
    opt_states = opti.variable(N+1, 3)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]
    # opti.set_initial(opt_states, np.zeros((N+1, 3)))
    # parameters
    opt_x0 = opti.parameter(3)
    opt_xs = opti.parameter(3)
    # create model
    f = lambda x_, u_: ca.vertcat(*[u_[0]*np.cos(x_[2]), u_[0]*np.sin(x_[2]), u_[1]])
    f_np = lambda x_, u_: np.array([u_[0]*np.cos(x_[2]), u_[0]*np.sin(x_[2]), u_[1]])

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :]==x_next)

    ## define the cost function
    ### some addition parameters
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, .1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :]-opt_xs.T), Q, (opt_states[i, :]-opt_xs.T).T]) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])

    opti.minimize(obj)

    #### boundrary and control conditions
    opti.subject_to(opti.bounded(-2.0, x, 2.0))
    opti.subject_to(opti.bounded(-2.0, y, 2.0))
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))

    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    opti.solver('ipopt', opts_setting)
    final_state = np.array([1.5, 1.5, 0.0])
    opti.set_value(opt_xs, final_state)

    t0 = 0
    init_state = np.array([0.0, 0.0, 0.0])
    current_state = init_state.copy()
    next_states = np.zeros((N+1, 3))
    u0 = np.zeros((N, 2))
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0

    while(np.linalg.norm(current_state-final_state)>1e-2 and mpciter-sim_time/T<0.0  ):
        ## set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x0, current_state)
        print((next_states))
        opti.set_initial(opt_controls, u0.reshape(N, 2))
        opti.set_initial(opt_states, next_states.reshape(N+1, 3))
        ## solve the problem once again
        sol = opti.solve()
        ## obtain the control input
        u = sol.value(opt_controls)
        u_c.append(u[0, :])
        t_c.append(t0)
        next_states = prediction_state(x0=current_state, u=u, N=N, T=T)
        x_c.append(next_states)
        t0, current_state, u0, next_states = shift_movement(T, t0, current_state, u, next_states, f_np)
        mpciter = mpciter + 1
        xx.append(current_state)

    ## after loop
    print(mpciter)
    print('final error {}'.format(np.linalg.norm(final_state-current_state)))
    ## draw function
    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=init_state, target_state=final_state,robot_states=xx)
