#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
from draw import Draw_MPC_point_stabilization_v1, draw_gt, draw_gt_measurements, draw_gtmeas_noisemeas, draw_gt_mhe_measurements

def shift_movement(T, t0, x0, u, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T*f_value
    t = t0 + T
    u_end = ca.horzcat(u[:, 1:], u[:, -1])

    return t, st, u_end

def structure_result(data, n_c=2, n_s=3,):
    temp_1 = data[:-n_s].reshape(-1, n_c+n_s)
    # print(temp_1)
    u_ = temp_1[:, :n_c].T
    s_ = temp_1[:, n_c:].T
    s_ = np.concatenate((s_, data[-n_s:].reshape(n_s, 1)), axis=1)
    return u_, s_ # output in shape (n, N)

def shift_trajectory(state, u):
    ##########################################################
    ## state and u in form (state, N)
    state_ = np.concatenate((state.T[1:], state.T[-1:]))
    u_ = np.concatenate((u.T[1:], u.T[-1:]))
    return u_, state_

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
    x0_ = x0.copy()
    xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1) # final state
    u0 = np.array([1,2]*N).reshape(-1, 2).T# np.ones((N, 2)) # controls
    ff_value = np.array([0.0, 0.0, 0.0]*(N+1)).reshape(-1, 3).T
    x_c = [] # contains for the history of the state
    u_c = []
    # t_c = [t0] # for the time
    t_c = [] # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0
    ### inital test
    c_p = current_parameters(0)
    init_control = optimizing_target(0)
    # print(u0.shape) u0 should have (n_controls, N)
    while(np.linalg.norm(x0-xs)>1e-2 and mpciter-sim_time/T<0.0 ):
        ## set parameter
        c_p['P'] = np.concatenate((x0, xs))
        init_control['X', lambda x:ca.horzcat(*x)] = ff_value 
        init_control['U', lambda x:ca.horzcat(*x)] = u0[:, 0:N]
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        estimated_opt = res['x'].full() # the feedback is in the series [u0, x0, u1, x1, ...]
        ff_last_ = estimated_opt[-3:]
        temp_estimated = estimated_opt[:-3].reshape(-1, 5)
        u0 = temp_estimated[:, :2].T
        ff_value = temp_estimated[:, 2:].T
        ff_value = np.concatenate((ff_value, estimated_opt[-3:].reshape(3, 1)), axis=1) # add the last estimated result now is n_states * (N+1)
        x_c.append(ff_value)
        u_c.append(u0[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u0, f)
        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        mpciter = mpciter + 1

    # draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0_, target_state=xs, robot_states=xx )

    # convert list data to numpy array 
    xx_np = np.array(xx)
    # draw_gt(t_c, xx_np)
    
    # synthesize the measurements
    con_cov = np.diag([0.005, np.deg2rad(2)])**2
    meas_cov = np.diag([0.1, np.deg2rad(2)])**2

    r = []
    alpha = []
    for i in range(len(xx)):
        r.append(np.sqrt(xx_np[i, 0]**2+xx_np[i, 1]**2) + np.sqrt(meas_cov[0, 0])*np.random.randn())
        alpha.append(np.arctan(xx_np[i, 1]/xx_np[i, 0]) + np.sqrt(meas_cov[1, 1])*np.random.randn())
    y_measurements = np.concatenate((np.array(r).reshape(-1, 1), np.array(alpha).reshape(-1, 1)), axis=1)
    # draw_gt_measurements(t_c, xx_np, y_measurements)
    # draw_gtmeas_noisemeas(t_c, xx_np, y_measurements)

    ## MHE 
    u_c_np = np.array(u_c)
    T_mhe = 0.2
    N_MHE = 6 # estimation horizon
    print("MHE horizon {}".format(N_MHE))
    ### using the same model and states （states, f, next_controls)
    mhe_target = ca_tools.struct_symSX([
        (
            ca_tools.entry('mhe_U', repeat=N_MHE, struct=controls),
            ca_tools.entry('mhe_X', repeat=N_MHE+1, struct=states)
        )
    ])
    mhe_U, mhe_X, = mhe_target[...] # data are stored in list [], notice that ',' cannot be missed

    mhe_measure_struct = ca_tools.struct_symSX([
        (
            ca_tools.entry('r'),
            ca_tools.entry('alpha')
        )
    ])
    mhe_r, mhe_alpha, = mhe_measure_struct[...]

    mhe_params = ca_tools.struct_symSX([
        (
            ca_tools.entry('Mes_ref', repeat=N_MHE+1, struct=mhe_measure_struct),
            ca_tools.entry('U_ref', repeat=N_MHE, struct=controls)
        )
    ])
    Mes_ref, U_ref, = mhe_params[...]

    ## measurement model
    f_m = lambda x, y: ca.vertcat(*[np.sqrt(x**2+y**2), np.arctan(y/x)])
    V_mat = np.linalg.inv(np.sqrt(meas_cov))
    W_mat = np.linalg.inv(np.sqrt(con_cov))
    #### cost function
    obj_mhe = 0 #### cost
    #### constrains
    g = [] # equal constrains
    #### multiple shooting constraints
    for i in range(N_MHE):
        x_next_ = f(mhe_X[i], mhe_U[i])*T + mhe_X[i]
        g.append(mhe_X[i+1] - x_next_)

    ### cost function for measurement 
    for i in range(N_MHE+1):
        h_x = f_m(mhe_X[i][0], mhe_X[i][1])
        temp_diff_ = Mes_ref[i] - h_x
        obj_mhe = obj_mhe + ca.mtimes([temp_diff_.T, V_mat, temp_diff_])
    for i in range(N_MHE):
        temp_diff_ = U_ref[i] - mhe_U[i]
        obj_mhe = obj_mhe + ca.mtimes([temp_diff_.T, W_mat, temp_diff_])

    ### define MHE nlp problem, the constraints stay the same as MPC
    nlp_prob_mhe = {'f': obj_mhe, 'x': mhe_target, 'p':mhe_params, 'g':ca.vertcat(*g)}
    mhe_opts_setting = {'ipopt.max_iter':2000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    mhe_solver = ca.nlpsol('solver', 'ipopt', nlp_prob_mhe, mhe_opts_setting)
    
    mhe_lbg = 0.0
    mhe_ubg = 0.0
    mhe_lbx = []
    mhe_ubx = []

    ## add constraints to control and statesn notice that for the N+1 th state
    for _ in range(N_MHE):
        mhe_lbx.append(-v_max)
        mhe_lbx.append(-omega_max)
        mhe_ubx.append(v_max)
        mhe_ubx.append(omega_max)
        mhe_lbx.append(-2.0)
        mhe_lbx.append(-2.0)
        mhe_lbx.append(-np.inf)
        mhe_ubx.append(2.0)
        mhe_ubx.append(2.0)
        mhe_ubx.append(np.inf)
    # for the N+1 state
    mhe_lbx.append(-2.0)
    mhe_lbx.append(-2.0)
    mhe_lbx.append(-np.inf)
    mhe_ubx.append(2.0)
    mhe_ubx.append(2.0)
    mhe_ubx.append(np.inf)


    ## MHE simulation
    X0 = np.zeros((N_MHE+1, 3))
    U0 = np.array(u_c[:N_MHE])
    X_estimate = None
    U_estimate = None
    for i in range(N_MHE+1):
        X0[i] = np.array([y_measurements[i, 0]*np.cos(y_measurements[i, 1]),
        y_measurements[i, 0]*np.sin(y_measurements[i, 1]),
                                    0.0])
    init_mhe_state = mhe_target(0)
    init_mhe_params = mhe_params(0)

    for i in range(y_measurements.shape[0]-N_MHE):
        mheiter = i
        init_mhe_state['mhe_U', lambda x:ca.horzcat(*x)] = U0.T # input size should be (controls, N_MHE)
        init_mhe_state['mhe_X', lambda x:ca.horzcat(*x)] = X0.T 
        init_mhe_params['Mes_ref', lambda x:ca.horzcat(*x)] = y_measurements[i:i+N_MHE+1].T
        init_mhe_params['U_ref', lambda x:ca.horzcat(*x)] = np.array(u_c[i:i+N_MHE]).T
        mhe_res = mhe_solver(x0=init_mhe_state, p=init_mhe_params, lbg=mhe_lbg, lbx=mhe_lbx, ubg=mhe_ubg, ubx=mhe_ubx)
        mhe_estimated = mhe_res['x'].full()

        u_sol, state_sol = structure_result(mhe_estimated)

        if U_estimate is None:
            U_estimate = u_sol.T[N_MHE-1:].reshape(1, -1)
        else:
            U_estimate = np.concatenate((U_estimate, u_sol.T[N_MHE-1].reshape(1, -1)))
        if X_estimate is None:
            X_estimate = state_sol.T[N_MHE:].reshape(1, -1)
        else:
            X_estimate = np.concatenate((X_estimate, state_sol.T[N_MHE:].reshape(1, -1)))
        # shift trajectories to initialize the next step
        U0, X0 = shift_trajectory(state_sol, u_sol)
    draw_gt_mhe_measurements(t_c, xx_np, y_measurements, X_estimate, n_mhe=N_MHE)