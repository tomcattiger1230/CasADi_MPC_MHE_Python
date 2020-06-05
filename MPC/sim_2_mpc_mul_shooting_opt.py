#!/usr/bin/env python
# coding=utf-8

import casadi as ca
import numpy as np

from draw import Draw_MPC_point_stabilization_v1

def shift_movement(T, t0, x0, u, f):
    f_value = f_np(x0, u[0])
    st = x0 + T*f_value
    t = t0 + T
    u_end = np.concatenate((u[:, 1:], u[:, -1:]))

    return t, st, u_end

def prediction_state(x0, u, T, N):
    # define predition horizon function
    states = np.zeros((N+1, 3))
    states[0, :] = x0
    for i in range(N):
        states[i+1, 0] = states[i, 0] + u[i, 0] * np.cos(states[i, 2]) * T
        states[i+1, 1] = states[i, 1] + u[i, 0] * np.sin(states[i, 2]) * T
        states[i+1, 2] = states[i, 2] + u[i, 1] * T
    return states

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
    f_np = lambda x, u: np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])

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

    while(np.linalg.norm(current_state-final_state)>1e-2 and mpciter-sim_time/T<0.0  ):
        ## set parameter, here only update initial state of x (x0)
        opti.set_value(x0, current_state)
        ## solve the problem once again
        sol = opti.solve()
        ## obtain the control input
        u = sol.value(controls)
        u_c.append(u[0, :])
        t_c.append(t0)
        next_states = prediction_state(x0=current_state, u=u, N=N, T=T)
        x_c.append(next_states)
        t0, current_state, u0 = shift_movement(T, t0, current_state, u, f)
        mpciter = mpciter + 1
        xx.append(current_state)

    ## after loop
    print(mpciter)
    print('final error {}'.format(np.linalg.norm(final_state-current_state)))
    ## draw function
    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=init_state, target_state=final_state,robot_states=xx)
