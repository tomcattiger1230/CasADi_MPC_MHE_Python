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

    # X = ca.SX.sym('X', n_states, N+1)

    current_parameters = ca_tools.struct_symSX([
        (
            ca_tools.entry('P', shape=n_states+n_states),
        )
    ])
    P, = current_parameters[...]

    ### define
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, .1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    #### constrains
    g = [] # equal constrains
    g.append(X[0]-P[:3]) # initial condition constraints
    for i in range(N):
        obj = obj + ca.mtimes([(X[i]-P[3:]).T, Q, X[i]-P[3:]]) + ca.mtimes([U[i].T, R, U[i]])
        x_next_ = f(X[i], U[i])*T + X[i]
        g.append(X[i+1] - x_next_)



    nlp_prob = {'f': obj, 'x': optimizing_target, 'p':current_parameters, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':200, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # lbg = 0.0
    # ubg = 0.0
    # lbx = []
    # ubx = []

    # ## add constraints to control and statesn notice that for the N+1 th state
    # for _ in range(N):
    #     lbx = lbx + [-v_max, -omega_max, -2.0, -2.0, -np.inf]
    #     ubx = ubx + [v_max, omega_max, 2.0, 2.0, np.inf]
    #     # lbx.append(-v_max)
    #     # lbx.append(-omega_max)
    #     # ubx.append(v_max)
    #     # ubx.append(omega_max)
    #     # lbx.append(-2.0)
    #     # lbx.append(-2.0)
    #     # lbx.append(-np.inf)
    #     # ubx.append(2.0)
    #     # ubx.append(2.0)
    #     # ubx.append(np.inf)
    # # for the N+1 state
    # lbx.append(-2.0)
    # lbx.append(-2.0)
    # lbx.append(-np.inf)
    # ubx.append(2.0)
    # ubx.append(2.0)
    # ubx.append(np.inf)

    lbg = 0.0
    ubg = 0.0
    lbx = optimizing_target(-ca.inf)
    ubx = optimizing_target(ca.inf)
    lbx['U', :, 'v'] = -v_max
    lbx['U', :, 'omega'] = -omega_max
    lbx['X', :, 'x'] = -2.0
    lbx['X', :, 'y'] = -2.0
    ubx['U', :, 'v'] = v_max
    ubx['U', :, 'omega'] = omega_max
    ubx['X', :, 'x'] = 2.0
    ubx['X', :, 'y'] = 2.0

    # Simulation
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)# initial state
    x0_ = x0.copy()
    xs = np.array([1.5, 1.5, np.pi/2.0]).reshape(-1, 1) # final state
    u0 = np.array([1,2]*N).reshape(-1, 2).T# np.ones((N, 2)) # controls
    ff_value = np.array([0.0, 0.0, 0.0]*(N+1)).reshape(-1, 3).T
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
    # print(u0.shape) u0 should have (n_controls, N)
    while(np.linalg.norm(x0-xs)>1e-2 and mpciter-sim_time/T<0.0 ):
        ## set parameter
        c_p['P'] = np.concatenate((x0, xs))
        init_control['X', lambda x:ca.horzcat(*x)] = ff_value
        init_control['U', lambda x:ca.horzcat(*x)] = u0[:, 0:N]
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time()- t_)
        estimated_opt = res['x'].full() # the feedback is in the series [u0, x0, u1, x1, ...]
        ff_last_ = estimated_opt[-3:]
        temp_estimated = estimated_opt[:-3].reshape(-1, 5)
        u0 = temp_estimated[:, :2].T
        ff_value = temp_estimated[:, 2:].T
        ff_value = np.concatenate((ff_value, estimated_opt[-3:].reshape(3, 1)), axis=1) # add the last estimated result now is n_states * (N+1)
        # print(ff_value.T)
        x_c.append(ff_value)
        u_c.append(u0[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u0, f)
        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        mpciter = mpciter + 1
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))
    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0_, target_state=xs, robot_states=xx )
