#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
from draw import Draw_MPC_tracking

def shift_movement(T, t0, x0, u, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T*f_value
    t = t0 + T
    u_end = ca.horzcat(u[:, 1:], u[:, -1])

    return t, st, u_end

def desired_trajectory(current_time_, x0_, N_):
    # initial pose
    p_ = x0_.reshape(1, -1).tolist()[0]
    # trajectory for next N steps
    for i in range(N_):
        t_predict = current_time_ + i*T
        x_ref_ = 0.5 * t_predict
        y_ref_ = 1.0
        theta_ref_ = 0.0
        if x_ref_ >= 12.0:
            x_ref_ = 12.0
            y_ref_ = 1.0
            theta_ref_ = 0.0
        p_.append(x_ref_)
        p_.append(y_ref_)
        p_.append(theta_ref_)

    return np.array(p_).reshape(N_+1, -1).T

def desired_controls(current_time_, N_):
    p_ = []
    for i in range(N_):
        t_predict = current_time_ + i*T
        x_ref_ = 0.5 * t_predict
        v_ref_ = 0.5
        omega_ref_ = 0.0
        if x_ref_ >= 12.0:
            v_ref_ = 0.0
            omega_ref_ = 0.0
        p_.append(v_ref_)
        p_.append(omega_ref_)
    return np.array(p_).reshape(N_, -1).T

def get_estimated_result(data, N_):
    x_ = np.zeros((N_+1, 3))
    u_ = np.zeros((N_, 2))
    print(data[:3])
    x_[0] = data[:3].T#.reshape(-1, 1)
    for i in range(N_):
        x_[i+1] = data[3+i*5:6+i*5].T#.reshape(-1, 1)
        u_[i] = data[6+5*i:8+5*i].T#.reshape(-1, 1)
    return x_.T, u_.T



if __name__ == '__main__':
    T = 0.5 # sampling time [s]
    N = 8 # prediction horizon
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
    rhs['x'] = v*np.cos(theta)
    rhs['y'] = v*np.sin(theta)
    rhs['theta'] = omega

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    optimizing_target = ca_tools.struct_symSX([
        (
            ca_tools.entry('U', repeat=N, struct=controls),
            ca_tools.entry('X', repeat=N+1, struct=states)
        )
    ])
    U, X, = optimizing_target[...] # data are stored in list [], notice that ',' cannot be missed

    ### basically here are the parameters that for trajectory definition
    current_parameters = ca_tools.struct_symSX([
        (
            ca_tools.entry('U_ref', repeat=N, struct=controls),
            ca_tools.entry('X_ref', repeat=N+1, struct=states)
        )
    ])
    U_ref, X_ref,  = current_parameters[...]

    ### define
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, .5]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    #### constrains
    g = [] # equal constrains
    g.append(X[0]-X_ref[0]) # initial condition constraints
    for i in range(N):
        # state_error = X[i] - P[i*5+3:i*5+6]
        # control_error = U[i] - P[i*5+6:i*5+8]
        state_error_ = X[i] - X_ref[i]
        control_error_ = U[i] - U_ref[i]
        obj = obj + ca.mtimes([state_error_.T, Q, state_error_]) + ca.mtimes([control_error_.T, R, control_error_])
        x_next_ = f(X[i], U[i])*T + X[i]
        g.append(X[i+1] - x_next_)

    nlp_prob = {'f': obj, 'x': optimizing_target, 'p':current_parameters, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':2000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    lbg = 0.0
    ubg = 0.0
    lbx = []
    ubx = []

    ## add constraints to control and statesn notice that for the N+1 th state
    for _ in range(N):
        lbx.append(-v_max)
        lbx.append(-omega_max)
        ubx.append(v_max)
        ubx.append(omega_max)
        lbx.append(-20.0)
        lbx.append(-2.0)
        lbx.append(-np.inf)
        ubx.append(20.0)
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
    x0_ = x0.copy()
    u0 = np.array([0.0, 0.0]*N).reshape(-1, 2).T# np.ones((N, 2)) # controls
    ff_value = np.array([0.0, 0.0, 0.0]*(N+1)).reshape(-1, 3).T
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    sim_time = 20.0
    xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1) # final state
    ## start MPC
    mpciter = 0
    ### inital test
    c_p = current_parameters(0) # references
    init_input = optimizing_target(0)
    # print(u0.shape) u0 should have (n_controls, N)
    while(mpciter-sim_time/T<0.0 and mpciter<50):
        current_time = mpciter * T # current time
        ## obtain the desired trajectory
        a = desired_controls(current_time, N)
        print('trajectory {}'.format(a))
        c_p['X_ref', lambda x:ca.horzcat(*x)] = desired_trajectory(current_time, x0, N)
        c_p['U_ref', lambda x:ca.horzcat(*x)] = desired_controls(current_time, N)
        ## set parameter
        init_input['X', lambda x:ca.horzcat(*x)] = ff_value
        init_input['U', lambda x:ca.horzcat(*x)] = u0[:, 0:N]
        res = solver(x0=init_input, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        estimated_opt = res['x'].full() # the feedback is in the series [u0, x0, u1, x1, ...]
        print(estimated_opt.shape)
        ff_value, u0 = get_estimated_result(estimated_opt, N)
        # u0 = temp_estimated[:, :2].T
        # ff_value = temp_estimated[:, 2:].T
        # ff_value = np.concatenate((ff_value, estimated_opt[-3:].reshape(3, 1)), axis=1) # add the last estimated result now is n_states * (N+1)
        print(ff_value.T)
        x_c.append(ff_value)
        u_c.append(u0[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u0, f)
        x0 = ca.reshape(x0, -1, 1)
        x0 = x0.full()
        xx.append(x0)
        mpciter = mpciter + 1
    print(mpciter)
    draw_result = Draw_MPC_tracking(rob_diam=0.3, init_state=x0_, robot_states=xx )
