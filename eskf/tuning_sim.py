import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_sim = ESKFTuningParams(
    accm_std=1e-4,# 0.07 / (3600**0.5),
    accm_bias_std= 1e-3, #0.07 / (3600**0.5),#1.89 / 160**1.5,
    accm_bias_p=1/(3600*3),#1e-11,

    gyro_std=1e-5, #0.15 / (3600**0.5),
    gyro_bias_std=1e-6,# 0.15 / (3600**0.5)/10,#0.9 / 3600**1.5,# 4.16e-6,
    gyro_bias_p=1/ (3600*3),#1e-11,

    gnss_std_ne=0.21,
    gnss_std_d=0.4)

x_nom_init_sim = NominalState(
    np.array([20.0, 0., 0.]),  # position
    np.array([1.0, 0., -1.0]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[1.,  # position
                            1.,  # velocity
                            np.deg2rad(1),  # angle vector
                            1.5e-2,  # accelerometer bias
                            1.5e-4])  # gyro bias
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)
