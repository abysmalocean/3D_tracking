from __future__ import print_function, division
from builtins import range
import numpy as np
from math import sin, sqrt, cos
"""
This file defines layers types that are commonly used for EKF_Net
"""

def efk_net_predict_forward(x_post, p_post, dt, predict ,F_gen, B_gen, Q_noise):
    """
        Prediction forward
    """
    x_hat = predict(x_post, dt)
    F     = F_gen(x_post, dt)
    B     = B_gen(x_post, dt)
    BQ    = np.dot(B, Q_noise)
    BQB   = np.dot(BQ, B.transpose())
    FP    = np.dot(F, p_post)
    FPFt  = np.dot(FP, F.transpose())
    p_hat = FPFt + BQB
    cache = (x_hat, F, BQ, BQB, FP, FPFt, p_hat)
    return x_hat, p_hat, cache

    print("This is the forward prediction function")


def ekf_net_update_forward(x_hat, p_hat, gen_h, gen_H, R, z): 
    
    n = x_hat.shape[0]
    H = gen_H(x_hat)

    y_ = np.array([z[0], 
                   z[1],
                   z[2]])[:, None] - gen_h(x_hat)
    
    y_.itemset(2, angle_difference(z[2], (gen_h(x_hat).item(2)) % (2 * np.pi)))

    PHT = np.dot(p_hat, H.transpose())  
    S   = np.dot(H, PHT) + R
    S_inv = np.linalg.inv(S)
    K   = np.dot(PHT, S_inv)
    KH     = np.dot(K, H)
    IKH    = np.identity(n) - KH
    p_post = np.dot(IKH, p_hat)
    x_post = x_hat + np.dot(K, y_)

    cache = (y_, PHT, S, S_inv, K, KH, IKH, x_post, x_hat)

    return x_post, p_post, cache

def fun_update_9_back(dx_post, x_hat, K, y_): 
    """
    x_post = x_hat + np.dot(K, y_)
    """
    dx_hat = dx_post
    dK     = np.dot(dx_post, y_.T)
    dy_    = np.dot(K.T, dx_post)
    return dx_hat, dK, dy_

def fun_update_9_forward(x_hat, K, y_):
    x_post = x_hat + np.dot(K, y_)
    cache = (x_hat, K, y_)
    return x_post, cache

def fun_update_8_back(dp_post, p_hat, K, H): 
    print("fun_8 back ")
    n = 5; 
    KH     = np.dot(K, H)
    IKH    = np.identity(n) - KH
    p_post = np.dot(IKH, p_hat)

    dp_hat = np.dot(IKH.T, dp_post)
    dIKH   = np.dot(dp_post, p_hat.T)
    dK     = -np.dot(dIKH, H.T)
    dH     = -np.dot(K.T, dIKH)
    return dp_hat, dK, dH

def fun_update_8_forward(p_hat, K, H): 
    n = 5; 
    KH     = np.dot(K, H)
    IKH    = np.identity(n) - KH
    p_post = np.dot(IKH, p_hat)

    cache = (p_hat, K, H, KH, IKH, p_post)
    return p_post, cache

def fun_update_7_forward(p_hat, H , S): 
    PHT   = np.dot(p_hat, H.transpose())
    S_inv = np.linalg.inv(S)
    K     = np.dot(PHT, S_inv)
    cache = (p_hat, H, S_inv, PHT, S_inv, K)
    return K, cache
    
def fun_update_7_back(dK, p_hat, H , S): 
    
    # forward
    PHT   = np.dot(p_hat, H.transpose())
    S_inv = np.linalg.inv(S)
    K     = np.dot(PHT, S_inv)
    
    # backWard
    dS_inv = np.dot(PHT.T, dK)
    dS = inverse_dev_back_3(dS_inv, S)
    d_PHT = np.dot(dK, S_inv.T)
    dP_hat = np.dot(d_PHT, H)
    dH    = (np.dot(p_hat.T, d_PHT)).T

    return dP_hat, dH, dS


def inverse_dev_forward(S):
    S_inv = np.linalg.inv(S)
    return S_inv

def inverse_dev_back_3(dS_inv, S): 
    S_inv = np.linalg.inv(S)
    S_inv_T = np.linalg.inv(S.T)
    tmp = -np.kron(S_inv_T, S_inv)
    flat = dS_inv.reshape(1,9)
    ds = (np.dot(tmp, flat.T)).reshape((3,3))
    return ds

def inverse_dev_back(dS_inv, S): 
    S_inv = np.linalg.inv(S)
    S_inv_T = np.linalg.inv(S.T)

    tmp = -np.kron(S_inv_T, S_inv)
    print(tmp)
    print(dS_inv)
    flat = dS_inv.reshape(1, 4)
    flat[0][0] += flat[0][2]
    flat[0][1] += flat[0][3]
    flat[0][2] = 0
    flat[0][3] = 0 
    print(flat)
    ds = (np.dot(tmp, flat.T)).reshape((2,2))
    return ds

def fun_update_6_forward(H, p_hat, R):
    PHT = np.dot(p_hat, H.transpose())  
    S   = np.dot(H, PHT) + R
    cache = (PHT, S, H, p_hat, R)
    return S, cache

def fun_update_6_back(dS, H, p_hat, R): 

    # Forward
    PHT = np.dot(p_hat, H.transpose())  
    S   = np.dot(H, PHT) + R

    # backward
    dR = dS
    dPHT = np.dot(H.T, dS)
    dP = np.dot(dPHT, H)
    dHT = np.dot(p_hat.T, dPHT)
    dH = np.dot(dS, PHT.T) + dHT.T

    return dH, dP, dR

def fun_update_6_2_forward(x, p_hat, R, gen_H):
    H = gen_H(x)
    PHT = np.dot(p_hat, H.transpose())  
    S   = np.dot(H, PHT) + R
    cache = (PHT, S, H, p_hat, R)
    return S, cache

def fun_update_6_2_back(dS, x, p_hat, R, gen_H, EKF_filter): 

    # Forward
    H = gen_H(x)
    PHT = np.dot(p_hat, H.transpose())  
    S   = np.dot(H, PHT) + R

    # backward
    dR = dS
    dPHT = np.dot(H.T, dS)
    dP = np.dot(dPHT, H)
    dHT = np.dot(p_hat.T, dPHT)
    dH = np.dot(dS, PHT.T) + dHT.T
    dx   = np.zeros(5)

    l = EKF_filter.imu_off_l
    alpha = EKF_filter.imu_alpha
    dx[2] +=  l * cos(x[2] - alpha) * dH[0,2]
    dx[2] +=  l * sin(x[2] - alpha) * dH[1,2]
    return dx, dP, dR

def fun_update_5_forward(z, x_hat ,gen_h, gen_H):
    y_ = np.array([z[0], 
                   z[1],
                   z[2]])[:, None] - gen_h(x_hat)
    
    y_.itemset(2, angle_difference(z[2], (gen_h(x_hat).item(2)) % (2 * np.pi)))
    
    cache = (z, x_hat)
    return y_, cache

def fun_update_5_back(dy, z, x_hat ,gen_h, gen_H):
    y_ = np.array([z[0], 
                   z[1],
                   z[2]])[:, None] - gen_h(x_hat)
    
    y_.itemset(2, angle_difference(z[2], (gen_h(x_hat).item(2)) % (2 * np.pi)))

    H = gen_H(x_hat)
    print(H)
    dx_hat = np.random.randn(5)
    dx_hat = -np.dot(H.T, dy).T

    return dx_hat

def fun_predict_2_forward(Q, x_post, p_post, dt ,F_gen): 
    F     = F_gen(x_post, dt)
    FP    = np.dot(F, p_post)
    FPFt  = np.dot(FP, F.transpose())
    p_hat = np.add(FPFt, Q)
    cache = []
    #cache = (F, FP, FPFt, p_hat, x_post, p_post, dt)
    
    return p_hat, cache

def fun_predict_2_back(dp_pred, x_post, p_post, dt ,F_gen, Q, wheelbase): 
    # forward step
    F     = F_gen(x_post, dt)
    FP    = np.dot(F, p_post)
    FPFt  = np.dot(FP, F.transpose())
    p_hat = FPFt + Q

    dq = dp_pred
    dx_post = np.zeros(5)
    dp_post = np.random.randn(5,5)

    dFP = np.dot(dp_pred, F)
    dp_post = np.dot(F.T, dFP)

    dF_1 = (np.dot(FP.T, dp_pred)).T
    dF_2 = np.dot(dFP, p_post.T)
    dF = dF_1 + dF_2
    print(dF)
    state = x_post
    x = state.item(0)
    y = state.item(1)
    th = state.item(2)
    v = state.item(3)
    phi = state.item(4)
    # dF[0,2]
    dth_F02  = -cos(th) * cos(phi) * v * dt * dF[0,2]
    dphi_F02 =  sin(th) * sin(phi) * v * dt * dF[0,2]
    dv_F02   = -sin(th) * cos(phi) *     dt * dF[0,2]
    # dF[0,3]
    dth_F03  =  -sin(th) * cos(phi) * dt * dF[0,3]
    dphi_F03 =  -cos(th) * sin(phi) * dt * dF[0,3]
    dv_F03   =  0.0 
    # dF[0,4]
    dth_F04  =   sin(th) * sin(phi) * v * dt * dF[0,4]
    dphi_F04 =  -cos(th) * cos(phi) * v * dt * dF[0,4]
    dv_F04   =  -cos(th) * sin(phi)     * dt * dF[0,4]
    # dF[1, 2]
    dth_F12  =  -sin(th) * cos(phi) * v * dt * dF[1,2]
    dphi_F12 =  -cos(th) * sin(phi) * v * dt * dF[1,2]
    dv_F12   =   cos(th) * cos(phi)     * dt * dF[1,2]

    # dF[1, 3]
    dth_F13  =  cos(th) * cos(phi) * dt * dF[1,3]
    dphi_F13 = -sin(th) * sin(phi) * dt * dF[1,3]
    dv_F13   =  0.0

    # dF[1, 4]
    dth_F14  = -cos(th) * sin(phi) * v * dt * dF[1,4]
    dphi_F14 = -sin(th) * cos(phi) * v * dt * dF[1,4]
    dv_F14   = -sin(th) * sin(phi) *     dt * dF[1,4]

    # dF[2, 3]
    dth_F23  = 0.0
    dphi_F23 = (cos(phi) * dt / wheelbase) * dF[2,3]
    dv_F23   = 0.0

    # dF[2, 4]
    dth_F24  = 0.0
    dphi_F24 = (-sin(phi) * v * dt / wheelbase) * dF[2,4]
    dv_F24   = ( cos(phi)     * dt / wheelbase) * dF[2,4]

    dx_post[2] = dth_F02 + dth_F03 + dth_F04 + dth_F12 + dth_F13 + \
                 dth_F14 + dth_F23 + dth_F24 
    dx_post[4] = dphi_F02 + dphi_F03 + dphi_F04 + \
                 dphi_F12 + dphi_F13 + dphi_F14 + \
                 dphi_F23 + dphi_F24
    dx_post[3] = dv_F02 + dv_F03 + dv_F04 + \
                 dv_F12 + dv_F13 + dv_F14 +\
                 dv_F23 + dv_F24
    dx_post_tmp = np.random.randn(5,1)
    dx_post_tmp[0,0] = 0.0
    dx_post_tmp[1,0] = 0.0
    dx_post_tmp[2,0] = dx_post[2]
    dx_post_tmp[3,0] = dx_post[3]
    dx_post_tmp[4,0] = dx_post[4]

    return dx_post_tmp, dp_post, dq


def fun_predict_1_forward(x_post, predict, dt):
    x_hat = predict(x_post, dt)
    cache = (x_post, x_hat)
    return x_hat, cache

def fun_predict_1_back(dx_pred, x_post, predict, dt, F_gen): 
    F     = F_gen(x_post, dt)
    dx_post = np.dot(F.T ,dx_pred)
    return dx_post


def update_forward(x_pred, p_pred, H, S, y_): 
    PHT    = np.dot(p_pred, H.T)
    S_inv  = np.linalg.inv(S)
    K      = np.dot(PHT, S_inv)
    KH     = np.dot(K, H)
    IKH    = np.identity(5) - KH
    x_post = x_pred + np.dot(K, y_)
    p_post = np.dot(IKH, p_pred)
    cache = (x_pred, p_pred, H, S, y_, PHT, S_inv, K, KH, IKH)
    return x_post, p_post, cache

def update_backward(dx_post, dp_post, cache):
    '''
    need to return
        dx_pred
        dp_pred
        dH
        dS
        dy_
    '''
    x_pred, p_pred, H, S, y_, PHT, S_inv, K, KH, IKH = cache
    dx_pred = None
    dp_pred = None
    dH = None
    dS = None
    dy_ = None
    dx_pred = np.random.randn(5,1)
    dp_pred = np.random.randn(5,5)
    dH = np.random.randn(3,5)
    dS = np.random.randn(3,3)

    """
    1.
    x_post = x_hat + np.dot(K, y_)
    """
    # dx_pred final
    dx_pred = dx_post
    dK_1      = np.dot(dx_post, y_.T)
    # dy_final
    dy_     = np.dot(K.T, dx_post)

    """
    2. 
    P = (I - KH)P
    """
    dp_pred_2 = np.dot(IKH.T, dp_post)
    dIKH      = np.dot(dp_post, p_pred.T)
    dK_2      = -np.dot(dIKH, H.T)
    dH_2      = -np.dot(K.T, dIKH)
    
    dK = dK_1 + dK_2
    """
    3. 
    K = pHTS-1
    """
    dS_inv = np.dot(PHT.T, dK)
    S_inv_T = np.linalg.inv(S.T)
    tmp = -np.kron(S_inv_T, S_inv)
    flat = dS_inv.reshape(1,9)
    # dS Final
    dS = (np.dot(tmp, flat.T)).reshape((3,3))

    d_PHT = np.dot(dK, S_inv.T)
    
    dp_pred_3 = np.dot(d_PHT, H)
    dH_3    = (np.dot(p_pred.T, d_PHT)).T

    dH = dH_2 + dH_3
    dp_pred = dp_pred_2 + dp_pred_3

    return dx_pred, dp_pred, dH, dS, dy_

def measurement_forward(x_pred, 
                        p_pred, 
                        z, 
                        R, 
                        gen_H,
                        gen_h):
    
    H = gen_H(x_pred)
    PHT = np.dot(p_pred, H.transpose())  
    S   = np.dot(H, PHT) + R

    y_ = np.array([z[0], 
                   z[1],
                   z[2]])[:, None] - gen_h(x_pred)
    y_.itemset(2, angle_difference(z[2], (gen_h(x_pred).item(2)) % (2 * np.pi)))

    cache = (R, H, PHT, S, y_, x_pred, p_pred, z)
    return y_, S, H, cache

def measurement_backward(dy_, dS, dH, cache): 
    '''
    Need to return
    dx_pred
    dp_pred
    dR    
    '''
    R, H, PHT, S, y_, x_pred, p_pred, z = cache
    dR = dS
    dPHT = np.dot(H.T, dS)
    dp_pred = np.dot(dPHT, H)
    
    dHT = np.dot(p_pred.T, dPHT)
    dH += np.dot(dS, PHT.T) + dHT.T
    
    dx_pred   = np.zeros(5)
    """
    l = EKF_filter.imu_off_l
    alpha = EKF_filter.imu_alpha
    """
    l = 0.323882694814033
    alpha = 1.4157995848709557
    dx_pred[2] +=  l * cos(x_pred[2] - alpha) * dH[0,2]
    dx_pred[2] +=  l * sin(x_pred[2] - alpha) * dH[1,2]

    dx_pred = np.add(-np.dot(H.T, dy_).T, dx_pred)
    #print(dx_pred)
    return dx_pred, dp_pred, dR

def prediction_forward(x_post, p_post, dt, 
                       F_gen, predict ,B_gen, 
                       Q_acc, Q_other):
    '''
    x_pred = f(x_post)
    p_pred = FPF + Q
    Q = B * sigma * B + Q_other
    '''
    B = B_gen(x_post, dt)
    F = F_gen(x_post, dt)
    FP    = np.dot(F, p_post)
    FPFt  = np.dot(FP, F.transpose())
    
    QBT = np.dot(Q_acc, B.T)
    Q   = np.dot(B, QBT) + Q_other
    p_pred = np.add(FPFt, Q)
    x_pred = predict(x_post, dt)

    cache = (B, F, FP, FPFt, QBT, Q, x_post, p_post, dt, Q_acc, Q_other)

    return x_pred, p_pred, cache

def prediction_backward(dx_pred, dp_pred, cache):
    '''
    should output the deraitive
    dx_post : state 
    dp_post : cov
    dQ_acc  : acceleration
    dQ_other: process Noise
    '''

    B, F, FP, FPFt, QBT, Q, x_post, p_post, dt, Q_acc, Q_other = cache
    
    # Get dF
    dFP = np.dot(dp_pred, F)
    dF_1 = (np.dot(FP.T, dp_pred)).T
    dF_2 = np.dot(dFP, p_post.T)
    dF = dF_1 + dF_2

    # Get the dp_post
    dp_post = np.dot(F.T, dFP)
    dx_post = get_dx_post_int_prediction(x_post, dF, dt)

    # Get dQ
    dQ = dp_pred
    dQ_other = dQ
    dQBT      = np.dot(B.T, dQ)
    dQ_acc    = np.dot(dQBT, B)
    dBT       = np.dot(Q_acc.T, dQBT)
    dB        = np.dot(dQ, QBT.T) + dBT.T

    dx_post[2] += -0.5 * (dt**2) * sin(x_post[2]) * dB[0,0]
    dx_post[2] +=  0.5 * (dt**2) * cos(x_post[2]) * dB[1,0]

    ## From Eq1
    dx_post += np.dot(F.T ,dx_pred)


    return dx_post, dp_post, dQ_acc, dQ_other

def get_dx_post_int_prediction(x_post, dF, dt):
    dx_post = np.zeros(x_post.shape)
    wheelbase = 2.71
    state = x_post
    x = state.item(0)
    y = state.item(1)
    th = state.item(2)
    v = state.item(3)
    phi = state.item(4)
    # dF[0,2]
    dth_F02  = -cos(th) * cos(phi) * v * dt * dF[0,2]
    dphi_F02 =  sin(th) * sin(phi) * v * dt * dF[0,2]
    dv_F02   = -sin(th) * cos(phi) *     dt * dF[0,2]
    # dF[0,3]
    dth_F03  =  -sin(th) * cos(phi) * dt * dF[0,3]
    dphi_F03 =  -cos(th) * sin(phi) * dt * dF[0,3]
    dv_F03   =  0.0 
    # dF[0,4]
    dth_F04  =   sin(th) * sin(phi) * v * dt * dF[0,4]
    dphi_F04 =  -cos(th) * cos(phi) * v * dt * dF[0,4]
    dv_F04   =  -cos(th) * sin(phi)     * dt * dF[0,4]
    # dF[1, 2]
    dth_F12  =  -sin(th) * cos(phi) * v * dt * dF[1,2]
    dphi_F12 =  -cos(th) * sin(phi) * v * dt * dF[1,2]
    dv_F12   =   cos(th) * cos(phi)     * dt * dF[1,2]

    # dF[1, 3]
    dth_F13  =  cos(th) * cos(phi) * dt * dF[1,3]
    dphi_F13 = -sin(th) * sin(phi) * dt * dF[1,3]
    dv_F13   =  0.0

    # dF[1, 4]
    dth_F14  = -cos(th) * sin(phi) * v * dt * dF[1,4]
    dphi_F14 = -sin(th) * cos(phi) * v * dt * dF[1,4]
    dv_F14   = -sin(th) * sin(phi) *     dt * dF[1,4]

    # dF[2, 3]
    dth_F23  = 0.0
    dphi_F23 = (cos(phi) * dt / wheelbase) * dF[2,3]
    dv_F23   = 0.0

    # dF[2, 4]
    dth_F24  = 0.0
    dphi_F24 = (-sin(phi) * v * dt / wheelbase) * dF[2,4]
    dv_F24   = ( cos(phi)     * dt / wheelbase) * dF[2,4]

    dx_post[2] = dth_F02 + dth_F03 + dth_F04 + dth_F12 + dth_F13 + \
                 dth_F14 + dth_F23 + dth_F24 
    dx_post[4] = dphi_F02 + dphi_F03 + dphi_F04 + \
                 dphi_F12 + dphi_F13 + dphi_F14 + \
                 dphi_F23 + dphi_F24
    dx_post[3] = dv_F02 + dv_F03 + dv_F04 + \
                 dv_F12 + dv_F13 + dv_F14 +\
                 dv_F23 + dv_F24
    dx_post_tmp = np.random.randn(5,1)
    dx_post_tmp[0,0] = 0.0
    dx_post_tmp[1,0] = 0.0
    dx_post_tmp[2,0] = dx_post[2]
    dx_post_tmp[3,0] = dx_post[3]
    dx_post_tmp[4,0] = dx_post[4]

    return dx_post_tmp




def processNoise_forward(x_post, dt, B_gen, Q_acc, Q_other):
    B = B_gen(x_post, dt)
    QBT = np.dot(Q_acc, B.T)
    Q   = np.dot(B, QBT) + Q_other
    cache = (B, QBT, Q, x_post, dt, B_gen, Q_acc, Q_other)
    return Q, cache


def processNoise_backward(dQ, cache):
    '''
    return 
        dQ_acc
        dQ_other
        dx_post
    '''
    B, QBT, Q, x_post, dt, B_gen, Q_acc, Q_other = cache
    dQ_other  = dQ

    dQBT      = np.dot(B.T, dQ)
    dQ_acc    = np.dot(dQBT, B)
    dBT       = np.dot(Q_acc.T, dQBT)
    dB        = np.dot(dQ, QBT.T) + dBT.T

    dx_post = np.zeros(x_post.shape)
    dx_post[2] += -0.5 * (dt**2) * sin(x_post[2]) * dB[0,0]
    dx_post[2] +=  0.5 * (dt**2) * cos(x_post[2]) * dB[1,0]

    return dQ_acc, dQ_other, dx_post


def angle_difference(x, y):
    diff = (x - y) % (2 * np.pi)
    if diff > np.pi:
        return diff - (2 * np.pi)
    else:
        return diff

