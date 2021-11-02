import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_sim = ESKFTuningParams(
    accm_std=1e-4,
    accm_bias_std=4e-3,
    accm_bias_p=1/(3600*3),

    gyro_std=8.5e-5,
    gyro_bias_std=7e-6,
    gyro_bias_p=1/(3600*3),

    gnss_std_ne=0.4,
    gnss_std_d=0.7)

x_nom_init_sim = NominalState(
    np.array([0., 0., -5.0]),  # position
    np.array([20.0, 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[8e-2,  # position
                            8e-2,  # velocity
                            np.deg2rad(1),  # angle vector
                            8e-2,  # accelerometer bias
                            8e-4])  # gyro bias
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)
