import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_real = ESKFTuningParams(
    accm_std=1e-3,
    accm_bias_std=3e-4, #-3
    accm_bias_p=1/(3600*3),

    gyro_std=1e-3, #8.5
    gyro_bias_std=5e-4, #-5
    gyro_bias_p=1/(3600*3),

    gnss_std_ne=0.35,
    gnss_std_d=0.6,

    use_gnss_accuracy=False)

x_nom_init_real = NominalState(
    np.array([0., 0., 0.]),  # position
    np.array([0., 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., np.pi]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_real = np.repeat(repeats=3,  # repeat each element 3 times
                          a=[30.0,  # position
                             10.0,  # velocity
                             np.deg2rad(1),  # angle vector
                             8e-2,  # accelerometer bias
                             8e-4])  # gyro bias

x_err_init_real = ErrorStateGauss(np.zeros(15), np.diag(init_std_real**2), 0.)
